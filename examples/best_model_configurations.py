import os
from pathlib import Path

import pandas as pd


RESULT_STAT_SUFFIXES = ("_max", "_min", "_std", "_mean")
EXCLUDED_CONFIG_COLUMNS = {
    "healthy_installation_id",
    "prone_installation_id",
    "value",
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
    return str(value)


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
            "score": best_row[metric_column],
        }
    )


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
        .sort_values(["dataset", "model"])
        .drop(columns=["score"])
        .set_index(["dataset", "model"])
    )
    best_configurations.to_csv(cm_dir / "best_model_configurations.csv", index=True)

    with pd.option_context("display.max_colwidth", None):
        latex_table = best_configurations.to_latex(escape=True, multirow=True)
    (cm_dir / "best_model_configurations.tex").write_text(latex_table)

    print("Best model configurations - human readable")
    with pd.option_context("display.max_colwidth", None):
        print(best_configurations.to_string())
    print()
    print("Best model configurations - LaTeX")
    print(latex_table)


if __name__ == "__main__":
    main()
