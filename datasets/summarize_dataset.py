if __name__ == "__main__": import __config__

import argparse
import os
import re

import numpy as np
import pandas as pd


INSTALLATION_COLUMN_RE = re.compile(r"^(?P<installation>\d+)_(?P<measurement>.+)$")
SAMPLES_PER_HOUR = 12
DAYS_PER_YEAR = 365.25
MIN_PRODUCTION_SAMPLES_PER_DAY = 12
WINDOW_LEN_SEC = 60 * 5
SUMMARY_COLUMN_UNITS = {
    "Mounted Power": "kWp",
    "Avg. Year Prod.": "kWh",
}
RAW_COLUMNS = ["timestamp", "SourceId", "Power", "L1V", "L2V", "L3V"]
MOUNTED_POWER_BY_SOURCE_ID = {
    6: 9.9,
    159: 8.865,
    17: 7.7,
    87: 25,
}
WINTER_TIME = [
    ["2020-10-25", "2021-03-28"],
    ["2021-10-31", "2022-03-22"],
    ["2022-10-30", "2023-03-26"],
    ["2023-10-29", "2024-03-29"],
]


def load_wide_dataset(file_path):
    df = pd.read_csv(file_path, low_memory=False)
    if "timestamp" not in df.columns:
        raise ValueError("Expected a 'timestamp' column in dataset.csv")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    df.index.name = "timestamp"
    return df


def load_original_dataset(file_path):
    df = pd.read_csv(file_path, delimiter=",", header=None, names=RAW_COLUMNS)
    df = df.dropna().copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["SourceId"] = df["SourceId"].astype(int)

    for column in ["L1V", "L2V", "L3V"]:
        df.loc[df[column] < 210, column] = np.nan

    long_df = resample_installations(df)
    long_df["Grid OV Fault"] = long_df[["L1V", "L2V", "L3V"]].max(axis=1) > 253
    long_df["MountedPower"] = long_df["SourceId"].map(MOUNTED_POWER_BY_SOURCE_ID)

    long_df["timestamp"] = long_df["timestamp"] - pd.DateOffset(hours=2)
    for winter_time in WINTER_TIME:
        winter_mask = long_df["timestamp"].between(winter_time[0], winter_time[1])
        long_df.loc[winter_mask, "timestamp"] = long_df.loc[winter_mask, "timestamp"] + pd.DateOffset(hours=1)

    long_df.loc[long_df["Power"] > long_df["MountedPower"] * 1.2, "Power"] = 0
    long_df["NormalizedPower"] = long_df["Power"] / long_df["MountedPower"]
    long_df["timestamp unix"] = long_df["timestamp"].to_numpy(dtype="datetime64[s]").astype("int64")
    long_df["timestamp unix"] = long_df["timestamp unix"] - long_df["timestamp unix"].min()
    long_df["Bin"] = long_df["timestamp unix"] // WINDOW_LEN_SEC

    long_df = long_df.groupby(["SourceId", "Bin"], observed=True).agg({
        "timestamp": "min",
        "Power": "mean",
        "L1V": "max",
        "L2V": "max",
        "L3V": "max",
        "Grid OV Fault": "max",
        "MountedPower": "max",
        "NormalizedPower": "mean",
        "InstallationIndex": "max",
        "Id": "first",
    }).reset_index()

    return long_df


def resample_installations(df):
    unique_ids = df["SourceId"].unique()
    resampled_dfs = []
    for installation_index, source_id in enumerate(unique_ids):
        installation_df = df.loc[df["SourceId"] == source_id].copy()
        installation_df = installation_df.set_index("timestamp")
        installation_df = installation_df.resample("5min").agg({
            "Power": "mean",
            "L1V": "max",
            "L2V": "max",
            "L3V": "max",
        })

        installation_df["timestamp"] = installation_df.index
        installation_df["SourceId"] = source_id
        installation_df["InstallationIndex"] = installation_index
        installation_df["Id"] = f"PV {installation_index + 1}"
        installation_df["Power"] = installation_df["Power"].fillna(0)
        resampled_dfs.append(installation_df)

    return pd.concat(resampled_dfs, ignore_index=True)


def load_summary_input(file_path):
    header = pd.read_csv(file_path, nrows=0)
    has_wide_columns = "timestamp" in header.columns and any(
        INSTALLATION_COLUMN_RE.match(column) for column in header.columns
    )
    if has_wide_columns:
        return unpivot_installations(load_wide_dataset(file_path))

    return load_original_dataset(file_path)


def get_installation_columns(columns):
    installation_columns = {}
    for column in columns:
        match = INSTALLATION_COLUMN_RE.match(column)
        if match is None:
            continue

        installation = int(match.group("installation"))
        measurement = match.group("measurement")
        installation_columns.setdefault(installation, {})[measurement] = column

    return installation_columns


def unpivot_installations(df):
    installation_columns = get_installation_columns(df.columns)
    if not installation_columns:
        raise ValueError("No installation columns found. Expected names like '0_Power'.")

    long_dfs = []
    for installation in sorted(installation_columns):
        columns = installation_columns[installation]
        installation_df = df[list(columns.values())].rename(
            columns={source: measurement for measurement, source in columns.items()}
        )
        installation_df.insert(0, "timestamp", df.index)
        installation_df.insert(1, "InstallationIndex", installation)
        installation_df.insert(2, "Id", f"PV {installation + 1}")
        long_dfs.append(installation_df)

    return pd.concat(long_dfs, ignore_index=True)


def as_bool(series):
    if series.dtype == bool:
        return series.fillna(False)

    return series.fillna(False).astype(str).str.lower().isin(("true", "1", "yes"))


def summarize_installations(long_df):
    records = []
    for installation, installation_df in long_df.groupby("InstallationIndex", sort=True):
        samples = installation_df.loc[installation_df["Power"].notna()].copy()
        if samples.empty:
            continue

        power = pd.to_numeric(samples["Power"], errors="coerce").fillna(0.0)
        mounted_power = pd.to_numeric(samples["MountedPower"], errors="coerce").dropna().max()
        begin = samples["timestamp"].min()
        end = samples["timestamp"].max()

        grid_ov_fault = as_bool(samples["Grid OV Fault"])
        ovt = int(grid_ov_fault.sum())
        samples_count = int(len(samples))

        energy_kwh = power.sum() / SAMPLES_PER_HOUR
        production_days = count_production_days(samples["timestamp"], power)
        avg_year_production = np.nan
        if production_days > 0:
            avg_year_production = energy_kwh / production_days * DAYS_PER_YEAR

        indicator = np.nan
        expected_year_production = mounted_power * 1000
        if expected_year_production > 0:
            indicator = avg_year_production / expected_year_production

        records.append({
            "Id": f"PV {installation + 1}",
            "Prone factor": ovt / samples_count,
            r"\texttt{OVT}": ovt,
            "Samples count": samples_count,
            "Production days": production_days,
            "Mounted Power": mounted_power,
            "Begin": begin,
            "End": end,
            "Avg. Year Prod.": avg_year_production,
            "Indicator": indicator,
        })

    return pd.DataFrame(records).set_index("Id")


def count_production_days(timestamps, power):
    positive_samples_per_day = power.gt(0).groupby(timestamps.dt.date).sum()
    return int((positive_samples_per_day >= MIN_PRODUCTION_SAMPLES_PER_DAY).sum())


def filter_period(long_df, begin, end):
    period_df = long_df
    if begin:
        begin = pd.to_datetime(begin, dayfirst=True)
        period_df = period_df.loc[period_df["timestamp"] >= begin]
    if end:
        end = pd.to_datetime(end, dayfirst=True) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        period_df = period_df.loc[period_df["timestamp"] <= end]

    return period_df.copy()


def format_summary(summary):
    formatted = summary.copy()
    formatted["Prone factor"] = formatted["Prone factor"].map(lambda value: f"{value:.1%}")
    formatted["Mounted Power"] = formatted["Mounted Power"].map(lambda value: f"{value:g}")
    formatted["Begin"] = formatted["Begin"].dt.strftime("%d-%m-%Y")
    formatted["End"] = formatted["End"].dt.strftime("%d-%m-%Y")
    formatted["Avg. Year Prod."] = formatted["Avg. Year Prod."].map(lambda value: f"{value:.0f}")
    formatted["Indicator"] = formatted["Indicator"].map(lambda value: f"{value:.0%}")
    return formatted


def add_summary_row(formatted_summary, summary):
    ovt_column = r"\texttt{OVT}"
    summary_row = pd.Series("", index=formatted_summary.columns, name="Summary")
    summary_row["Prone factor"] = (
        f"{summary['Prone factor'].mean():.1%} $\\pm$ {summary['Prone factor'].std():.1%}"
    )
    summary_row[ovt_column] = f"{summary[ovt_column].mean():.0f}"
    summary_row["Indicator"] = f"{summary['Indicator'].mean():.1%} $\\pm$ {summary['Indicator'].std():.1%}"

    return pd.concat([formatted_summary, summary_row.to_frame().T])


def add_units_header(summary):
    summary_with_units = summary.copy()
    summary_with_units.columns = pd.MultiIndex.from_tuples(
        [(column, SUMMARY_COLUMN_UNITS.get(column, "")) for column in summary.columns]
    )
    return summary_with_units


def escape_latex_percent(summary):
    return summary.applymap(lambda value: value.replace("%", r"\%") if isinstance(value, str) else value)


def transpose_summary(summary):
    transposed = summary.T
    transposed.index = [
        f"{metric} [{unit}]" if unit else metric
        for metric, unit in transposed.index
    ]
    transposed.index.name = "Metric"
    return transposed


def remove_latex_horizontal_rules(latex):
    lines = latex.splitlines()
    return "\n".join(
        line for line in lines
        if line.strip() not in (r"\toprule", r"\midrule", r"\bottomrule")
    ) + "\n"


def parse_args():
    datasets_dir = os.path.dirname(os.path.realpath(__file__))
    repo_dir = os.path.dirname(datasets_dir)
    default_output_dir = os.path.join(repo_dir, "cm")
    parser = argparse.ArgumentParser(description="Prepare PV installation summary from dataset.csv.")
    parser.add_argument(
        "-i",
        "--input",
        default=os.path.join(datasets_dir, "orgdataset.csv"),
        help="Path to original orgdataset.csv file or already prepared wide dataset.csv file.",
    )
    parser.add_argument(
        "--csv",
        default=os.path.join(default_output_dir, "dataset_summary.csv"),
        help="Path where the formatted summary should be saved as CSV.",
    )
    parser.add_argument(
        "--latex",
        default=os.path.join(default_output_dir, "dataset_summary.tex"),
        help="Path where the formatted summary should be saved as LaTeX.",
    )
    parser.add_argument(
        "--begin",
        help="Optional first day included in calculations, in DD-MM-YYYY format.",
    )
    parser.add_argument(
        "--end",
        help="Optional last day included in calculations, in DD-MM-YYYY format.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    long_df = load_summary_input(args.input)
    period_df = filter_period(long_df, args.begin, args.end)
    summary = summarize_installations(period_df)
    formatted_summary = format_summary(summary)
    formatted_summary = add_summary_row(formatted_summary, summary)
    output_summary = add_units_header(formatted_summary)
    output_summary = transpose_summary(output_summary)

    print(f"Prepared rows: {len(long_df)}")
    if args.begin or args.end:
        print(f"Rows in selected period: {len(period_df)}")
    else:
        print("No period selected. Using each installation's full available period.")
    print(output_summary)
    print()
    latex_summary = escape_latex_percent(output_summary)
    column_format = "l" + "r" * len(latex_summary.columns)
    latex = latex_summary.to_latex(escape=False, column_format=column_format, multicolumn=False)
    latex = remove_latex_horizontal_rules(latex)
    print(latex)

    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        output_summary.to_csv(args.csv)
    if args.latex:
        os.makedirs(os.path.dirname(os.path.abspath(args.latex)), exist_ok=True)
        with open(args.latex, "w", encoding="utf-8") as file:
            file.write(latex)


if __name__ == "__main__":
    main()
