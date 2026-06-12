import os
import re
from pathlib import Path

import pandas as pd


RESULT_STAT_SUFFIXES = ("_max", "_min", "_std", "_mean")
EXCLUDED_CONFIG_COLUMNS = {
    "healthy_installation_id",
    "prone_installation_id",
    "value",
}
DATASET_MODEL_CONFIGURATION_FILENAME = "dataset_model_configuration.csv"
LEGACY_EXTERNAL_CONFIGURATION_FILENAME = "best_model_configurations.csv"
OUTPUT_CSV_FILENAME = "best_model_configurations_generated.csv"
OUTPUT_TEX_FILENAME = "best_model_configurations_generated.tex"
SCORE_COLUMN = "score"
DATASET_NAMES = {
    "db0": "PV1",
    "db1": "PV2",
    "db2": "PV3",
}
MODEL_NAMES = {
    "mlp": "$MLP_1$",
    "complex_mlp": "$MLP_2$",
    "nlastperiods": "LAGGED",
    "seaippf": "PROPOSED",
}
FLOAT_PATTERN = re.compile(r"^-?\d+(?:\.\d+)?(?:e[+-]?\d+)?$", re.IGNORECASE)
MULTIPLICATION_SUFFIX_PATTERN = re.compile(r"^(.+)\*(\d+(?:\.\d+)?)$")
LATEX_REPLACEMENTS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def is_config_column(column_name):
    """
    Return True when a result.csv column stores model configuration.
    """
    column_name = str(column_name)
    if column_name.startswith("Unnamed:"):
        return False
    if column_name in EXCLUDED_CONFIG_COLUMNS:
        return False
    return not column_name.endswith(RESULT_STAT_SUFFIXES)


def format_config_value(value):
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def latex_escape(value):
    value = str(value)
    return "".join(LATEX_REPLACEMENTS.get(char, char) for char in value)


def display_model_name(model):
    return MODEL_NAMES.get(model, str(model).upper())


def format_numeric_value(value):
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def normalize_config_parameter(name, value):
    name = name.strip()
    value = value.strip()
    if not FLOAT_PATTERN.match(value):
        return name, value

    numeric_value = float(value)
    multiplication_match = MULTIPLICATION_SUFFIX_PATTERN.match(name)
    if multiplication_match:
        name = multiplication_match.group(1)
        numeric_value *= float(multiplication_match.group(2))
    elif name.endswith("2^x"):
        name = name[: -len("2^x")]
        numeric_value = 2**numeric_value
    elif name.endswith("_10"):
        name = name[: -len("_10")]
        numeric_value *= 10

    return name, format_numeric_value(numeric_value)


def format_config_parameter(parameter):
    if "=" not in parameter:
        return parameter.strip()

    name, value = parameter.split("=", 1)
    name, value = normalize_config_parameter(name, value)
    return f"{name}={value}"


def format_config_parameter_for_latex(parameter):
    if "=" not in parameter:
        return latex_escape(parameter.strip())

    name, value = parameter.split("=", 1)
    name, value = normalize_config_parameter(name, value)

    latex_names = {
        "n": r"$n$",
        "n_step": r"$n\_step$",
    }
    latex_name = latex_names.get(name, latex_escape(name))
    return f"{latex_name}={latex_escape(value)}"


def split_config_parameters(config):
    if pd.isna(config) or not str(config).strip():
        return []
    return [format_config_parameter(parameter) for parameter in str(config).split(",")]


def chunk_parameters(parameters, chunk_size=3):
    return [
        parameters[index : index + chunk_size]
        for index in range(0, len(parameters), chunk_size)
    ]


def format_config_for_display(config):
    return "\n".join(
        ", ".join(chunk)
        for chunk in chunk_parameters(split_config_parameters(config))
    )


def format_config_for_latex(config):
    if pd.isna(config) or not str(config).strip():
        return ""
    parameters = [
        format_config_parameter_for_latex(parameter)
        for parameter in str(config).split(",")
    ]
    rows = [", ".join(chunk) for chunk in chunk_parameters(parameters)]
    if not rows:
        return ""
    return r"\makecell[l]{" + r" \\ ".join(rows) + "}"


def parse_dataset_model(result_dir):
    """
    Extract dataset and model from cm directory names like 20241201-140544_db2_nlastperiods_validation.
    """
    parts = result_dir.name.split("_")
    if parts and parts[-1] == "validation":
        parts = parts[:-1]
    if len(parts) >= 3:
        return parts[1], "_".join(parts[2:])
    if len(parts) == 2:
        return parts[0], parts[1]
    return result_dir.name, ""


def find_best_configuration(csv_path, metric_column="mae_mean"):
    """
    Load CSV file, select the row with the minimum metric value, and return its configuration.
    """
    df = pd.read_csv(csv_path)
    if metric_column not in df.columns:
        raise KeyError(f"Column '{metric_column}' not found in {csv_path}")

    best_idx = df[metric_column].idxmin()
    best_row = df.loc[best_idx]
    config_columns = [column for column in df.columns if is_config_column(column)]
    config = ", ".join(
        f"{column}={format_config_value(best_row[column])}"
        for column in config_columns
        if not pd.isna(best_row[column])
    )
    dataset, model = parse_dataset_model(Path(csv_path).parent)

    return pd.Series(
        {
            "dataset": dataset,
            "model": model,
            "config": config,
            SCORE_COLUMN: best_row[metric_column],
        }
    )


def normalize_configuration_headers(configuration):
    configuration = configuration.copy()
    configuration = configuration.loc[
        :, ~configuration.columns.astype(str).str.startswith("Unnamed:")
    ]
    configuration.columns = [str(column).strip().lower() for column in configuration.columns]

    deduplicated_configuration = pd.DataFrame(index=configuration.index)
    for column in dict.fromkeys(configuration.columns):
        column_values = configuration.loc[:, configuration.columns == column]
        if isinstance(column_values, pd.Series):
            deduplicated_configuration[column] = column_values
        else:
            deduplicated_configuration[column] = column_values.bfill(axis=1).iloc[:, 0]
    return deduplicated_configuration


def read_dataset_model_configuration(cm_dir):
    """
    Load optional per-dataset/model metadata used to extend the output table.
    """
    csv_paths = [
        cm_dir / DATASET_MODEL_CONFIGURATION_FILENAME,
        Path(DATASET_MODEL_CONFIGURATION_FILENAME),
        cm_dir / LEGACY_EXTERNAL_CONFIGURATION_FILENAME,
    ]
    csv_path = next((path for path in csv_paths if path.exists()), None)
    if csv_path is None:
        return pd.DataFrame()

    configuration = normalize_configuration_headers(pd.read_csv(csv_path))
    if {"dataset", "model"}.issubset(configuration.columns):
        return configuration.set_index(["dataset", "model"])

    configuration = normalize_configuration_headers(pd.read_csv(csv_path, index_col=[0, 1]))
    configuration.index.names = ["dataset", "model"]
    return configuration


def extend_with_dataset_model_configuration(best_configurations, cm_dir):
    dataset_model_configuration = read_dataset_model_configuration(cm_dir)
    if dataset_model_configuration.empty:
        return best_configurations

    extra_columns = [
        column
        for column in dataset_model_configuration.columns
        if column not in best_configurations.columns
    ]
    best_configurations = dataset_model_configuration.combine_first(best_configurations)
    ordered_columns = [
        column
        for column in ["config", SCORE_COLUMN, *extra_columns]
        if column in best_configurations.columns
    ]
    return best_configurations.reindex(columns=ordered_columns)


def sort_configurations_by_score(best_configurations):
    if SCORE_COLUMN not in best_configurations.columns:
        return best_configurations.sort_index()

    sortable = best_configurations.reset_index()
    sortable["_score_sort"] = pd.to_numeric(sortable[SCORE_COLUMN], errors="coerce")
    sortable = sortable.sort_values(
        ["dataset", "_score_sort", "model"],
        na_position="last",
    )
    return sortable.drop(columns=["_score_sort"]).set_index(["dataset", "model"])


def drop_internal_columns(best_configurations):
    return best_configurations.drop(columns=[SCORE_COLUMN], errors="ignore")


def capitalize_table_headers(table):
    table = table.copy()
    table.index.names = [
        str(name).capitalize() if name is not None else name
        for name in table.index.names
    ]
    table.columns = [str(column).capitalize() for column in table.columns]
    return table


def format_table_for_display(best_configurations, config_formatter):
    display_table = best_configurations.copy()
    if "config" in display_table.columns:
        display_table["config"] = display_table["config"].map(config_formatter)

    display_table.index = pd.MultiIndex.from_tuples(
        [
            (
                DATASET_NAMES.get(dataset, str(dataset).upper()),
                display_model_name(model),
            )
            for dataset, model in display_table.index
        ],
        names=display_table.index.names,
    )
    return capitalize_table_headers(display_table)


def human_readable_table_to_string(display_table):
    index_names = [name or "" for name in display_table.index.names]
    columns = [str(column) for column in display_table.columns]
    rows = []
    for (dataset, model), row in display_table.iterrows():
        cell_lines = {
            column: str(row[column]).splitlines() or [""]
            for column in display_table.columns
        }
        line_count = max(len(lines) for lines in cell_lines.values())
        for line_index in range(line_count):
            rows.append(
                [
                    str(dataset) if line_index == 0 else "",
                    str(model) if line_index == 0 else "",
                    *[
                        cell_lines[column][line_index]
                        if line_index < len(cell_lines[column])
                        else ""
                        for column in display_table.columns
                    ],
                ]
            )

    header = index_names + columns
    widths = [
        max(len(row[index]) for row in [header] + rows)
        for index in range(len(header))
    ]
    lines = [
        "  ".join(value.ljust(width) for value, width in zip(header, widths)).rstrip()
    ]
    lines.extend(
        "  ".join(value.ljust(width) for value, width in zip(row, widths)).rstrip()
        for row in rows
    )
    return "\n".join(lines)


def main():
    cm_dir = Path("cm")
    csv_files = []
    for file in os.listdir(cm_dir):
        csv_path = cm_dir / file / "result.csv"
        if csv_path.exists():
            csv_files.append(csv_path)

    best_configurations = pd.DataFrame(
        [find_best_configuration(csv_file) for csv_file in sorted(csv_files)]
    )
    best_configurations = (
        best_configurations.sort_values("score")
        .drop_duplicates(subset=["dataset", "model"], keep="first")
        .set_index(["dataset", "model"])
    )
    best_configurations = extend_with_dataset_model_configuration(best_configurations, cm_dir)
    best_configurations = sort_configurations_by_score(best_configurations)
    output_table = drop_internal_columns(best_configurations)
    capitalize_table_headers(output_table).to_csv(cm_dir / OUTPUT_CSV_FILENAME, index=True)

    human_readable_table = format_table_for_display(
        output_table,
        format_config_for_display,
    )
    latex_display_table = format_table_for_display(
        output_table,
        format_config_for_latex,
    )

    with pd.option_context("display.max_colwidth", None):
        latex_table = latex_display_table.to_latex(escape=False, multirow=True)
    (cm_dir / OUTPUT_TEX_FILENAME).write_text(
        "% Requires \\usepackage{makecell}\n" + latex_table
    )

    print("Best model configurations - human readable")
    print(human_readable_table_to_string(human_readable_table))
    print()
    print("Best model configurations - LaTeX")
    print(latex_table)


if __name__ == "__main__":
    main()
