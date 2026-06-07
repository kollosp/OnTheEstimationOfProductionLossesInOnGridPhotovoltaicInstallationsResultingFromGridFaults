import os

import pandas as pd
from pathlib import Path


def find_best_score(csv_path, column_name):
    """
    Load CSV file and find the row with the minimum value in a given column.
    
    Args:
        csv_path (str or Path): Path to the CSV file
        column_name (str): Name of the column to find minimum value in
    
    Returns:
        pd.Series: Row containing the minimum value
    """
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Find the index of the minimum value in the specified column
    min_idx = df[column_name].idxmin()
    
    # Get the row with the minimum value
    best_row = df[[col for col in df.columns if "mean" in col]].loc[[min_idx]]
    best_row.index = ["_".join(csv_path.parent.name.split("_")[1:])]  # Set the index to the model name extracted from the file path
    return best_row


def main():
    # Path to the result CSV file

    all_files = os.listdir("cm")

    csv_file = []
    for file in all_files:
        if os.path.exists(os.path.join("cm", file, "result.csv")):
            csv_file.append(Path("cm") / file / "result.csv")
                
    
    # Column to search for minimum value (change as needed)
    column = "mae_mean"  # You can change this to any column name
    
    # Find and display the best score

    best_score_files = sorted(
        set(Path("cm").glob("best_scrores*.csv")) | set(Path("cm").glob("best_scores*.csv"))
    )
    best_score_rows = []
    for best_score_file in best_score_files:
        best_score_row = pd.read_csv(best_score_file, index_col=0)
        best_score_row.index = best_score_row.index.astype(str)
        best_score_rows.append(best_score_row)

    best_rows = pd.concat(
        [find_best_score(csv_file, column) for csv_file in csv_file] + best_score_rows
    )
    best_rows = best_rows.sort_values("mae_mean")

    best_rows.to_csv(Path("cm") / "best_scores.csv", index=True)

    best_rows = best_rows.reset_index().rename(columns={"index": "case"})
    best_rows[["Dataset", "Model", "Subset"]] = best_rows["case"].str.extract(
        r"^([^_]+)_(.+?)(?:_(validation))?$"
    )
    best_rows["Subset"] = best_rows["Subset"].fillna("Hyperparameterization").replace(
        {"validation": "Validation"}
    )

    best_rows = best_rows.rename(
        columns={
            "mse_mean": "MSE",
            "mae_mean": "MAE",
            "rmse_mean": "RMSE",
            "max_error_mean": "ME",
        }
    )
    best_rows = best_rows.melt(
        id_vars=["Dataset", "Model", "Subset"],
        value_vars=["MSE", "MAE", "RMSE", "ME"],
        var_name="Metric",
        value_name="Score",
    )
    best_rows = best_rows.pivot_table(
        index=["Dataset", "Model"],
        columns=["Subset", "Metric"],
        values="Score",
        aggfunc="min",
    )
    best_rows = best_rows.reindex(
        columns=pd.MultiIndex.from_product(
            [["Hyperparameterization", "Validation"], ["MSE", "MAE", "RMSE", "ME"]]
        )
    )
    best_rows = pd.concat(
        [
            dataset_rows.sort_values(("Hyperparameterization", "MAE"))
            for _, dataset_rows in best_rows.groupby(level="Dataset", sort=True)
        ]
    )
    model_names = {
        "seaippf": "Proposed",
        "nlastperiods": "LAGGED",
        "complex_mlp": "$MLP_2$",
        "mlp": "$MLP_1$",
    }
    dataset_names = {
        "db0": "PV1",
        "db1": "PV2",
        "db2": "PV3",
    }
    best_rows.index = pd.MultiIndex.from_tuples(
        [
            (dataset_names.get(dataset, dataset), model_names.get(model, model.upper()))
            for dataset, model in best_rows.index
        ],
        names=best_rows.index.names,
    )

    best_rows_display = best_rows.copy().astype(object)
    for dataset in best_rows.index.get_level_values("Dataset").unique():
        dataset_rows = best_rows.loc[dataset]
        for metric_column in best_rows.columns:
            lowest_value = dataset_rows[metric_column].min()
            for model in dataset_rows.index:
                value = best_rows.loc[(dataset, model), metric_column]
                if pd.isna(value):
                    best_rows_display.loc[(dataset, model), metric_column] = ""
                    continue

                formatted_value = f"{value:.3f}".rstrip("0").rstrip(".")
                if formatted_value == "-0":
                    formatted_value = "0"
                delta = ((value - lowest_value) / lowest_value) * 100
                formatted_score = f"{formatted_value} ({delta:+.0f}%)"

                if value == lowest_value:
                    formatted_score = f"\\textbf{{{formatted_value}}}"
                best_rows_display.loc[(dataset, model), metric_column] = formatted_score

    print(f"Best scores for column \n{best_rows_display.to_string()}")

    best_rows_latex = best_rows_display.copy()
    best_rows_latex = best_rows_latex.applymap(lambda value: str(value).replace("%", r"\%"))
    best_rows_latex.index = pd.MultiIndex.from_tuples(
        [
            (
                dataset.replace("_", r"\_"),
                model if model.startswith("$") and model.endswith("$") else model.replace("_", r"\_"),
            )
            for dataset, model in best_rows_latex.index
        ],
        names=best_rows_latex.index.names,
    )
    latex_tables = []
    for subset in ["Hyperparameterization", "Validation"]:
        latex_tables.append(
            f"% {subset}\n"
            + best_rows_latex[subset].to_latex(escape=False, multirow=True)
        )
    Path("cm/best_scores.tex").write_text("\n\n".join(latex_tables))
    
    

    # print(f"Minimum value in '{column}': {best_row[column]}")
    # print("\nRow with minimum value:")
    # print(best_row)


if __name__ == "__main__":
    main()