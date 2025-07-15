from pathlib import Path

import pandas as pd


def process_dataframe(df: pd.DataFrame, columns: list[str], filename: Path) -> None:
    """
    Splits the 'name' column into table and column names and saves the resulting DataFrame to a CSV.

    :param df: DataFrame to process
    :param columns: List of DataFrame columns to extract
    :param filename: File path to save the output CSV
    """
    processed_df = df[columns].drop_duplicates().copy()

    # Create rows with separate table and column names
    processed_df[["table_name", "column_name"]] = processed_df[columns[0]].str.split(
        "-", expand=True
    )
    processed_df = processed_df[["table_name", "column_name", columns[1], columns[2]]]
    processed_df.columns = ["TableName", "ColumnName", "TableDesc", "ColumnDesc"]
    processed_df["ColumnType"] = ""

    # Save to CSV file
    processed_df.to_csv(filename, index=False)


def process_mappings(df: pd.DataFrame, filename: Path) -> None:
    """
    Processes mappings and saves them as CSV.

    :param df: DataFrame containing mapping data
    :param filename: File path to save the mappings CSV
    """
    mappings = df[df["label"] == 1][["omop", "table"]].drop_duplicates().copy()
    mappings[["TGT_ENT", "TGT_ATT"]] = mappings["omop"].str.split("-", expand=True)

    mappings[["SRC_ENT", "SRC_ATT"]] = mappings["table"].str.split("-", expand=True)
    mappings = mappings[["SRC_ENT", "SRC_ATT", "TGT_ENT", "TGT_ATT"]]
    mappings = mappings.groupby(["SRC_ENT", "SRC_ATT"]).agg(list).reset_index()
    mappings.to_parquet(filename, index=False)


def create_directories(paths: list[Path]) -> None:
    """
    Creates directories for the given paths if they don't exist.

    :param paths: List of directory paths to create
    """
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def main():
    # Define the base directory
    base_dir = Path(__file__).resolve().parent.parent

    # Set up directories
    raw_data_dir = base_dir / "raw_data" / "synthea-omop"
    source_dir = raw_data_dir / "source"
    target_dir = raw_data_dir / "target"
    mapping_dir = raw_data_dir / "mapping"

    # Define file paths
    input_file = raw_data_dir / "omop_synthea_data.xlsx"
    source_csv = source_dir / "synthea_schema.csv"
    target_csv = target_dir / "omop_schema.csv"
    mappings_parquet = mapping_dir / "synthea_to_omop_mapping.parquet"

    # Create directories if they don't exist
    create_directories([source_dir, target_dir, mapping_dir])

    # Read the Excel file
    df = pd.read_excel(input_file)

    # Process and save Source Data
    process_dataframe(df[["omop", "d1", "d2"]], ["omop", "d1", "d2"], target_csv)

    # Process and save Target Data
    process_dataframe(df[["table", "d3", "d4"]], ["table", "d3", "d4"], source_csv)

    # Process and save Mappings Data
    process_mappings(df, mappings_parquet)


if __name__ == "__main__":
    main()
