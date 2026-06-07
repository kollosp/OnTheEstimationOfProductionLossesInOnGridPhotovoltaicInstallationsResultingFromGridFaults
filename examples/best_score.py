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

    best_rows = pd.concat([find_best_score(csv_file, column) for csv_file in csv_file])

    print(f"Best scores for column \n{best_rows.to_string()}")
    
    

    # print(f"Minimum value in '{column}': {best_row[column]}")
    # print("\nRow with minimum value:")
    # print(best_row)


if __name__ == "__main__":
    main()