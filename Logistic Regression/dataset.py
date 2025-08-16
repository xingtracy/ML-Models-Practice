import os
import pandas as pd
import numpy as np
import kagglehub
import shutil

KAGGLE_DATASET = "miadul/kidney-disease-risk-dataset"
CSV_FILENAME = "kidney_disease_dataset.csv"  # CSV file name inside dataset
LOCAL_DIR = os.path.join(os.getcwd(), "kidney-disease-risk-dataset")
PARQUET_FILE = os.path.join(LOCAL_DIR, "Kidney_Disease_Risk_Performance.parquet")

def prepare_dataset():
    if not os.path.exists(LOCAL_DIR):
        print("Downloading dataset...")
        downloaded_path = kagglehub.dataset_download(KAGGLE_DATASET)
        shutil.copytree(downloaded_path, LOCAL_DIR)

    csv_path = os.path.join(LOCAL_DIR, CSV_FILENAME)

    # If Parquet exists, check if it's empty
    if os.path.exists(PARQUET_FILE):
        df = pd.read_parquet(PARQUET_FILE)
        if not df.empty:
            print("Loaded dataset from existing Parquet.")
            return df
        else:
            print("Parquet file is empty — rebuilding from CSV...")

    # Read from CSV and clean
    df = pd.read_csv(csv_path, na_values=["", " ", "NA", "N/A"])
    df = clean_dataframe(df)

    if df.empty:
        raise ValueError("No rows left after cleaning — check filtering rules.")

    df.to_parquet(PARQUET_FILE, index=False)
    print("Saved cleaned dataset to Parquet.")
    return df

def clean_dataframe(df):
    # Strip spaces from column names
    df.columns = df.columns.str.strip()

    # Map more possible missing values
    df.replace({
        "": np.nan,
        " ": np.nan,
        "na": np.nan,
        "n/a": np.nan,
        "nan": np.nan,
        "null": np.nan,
        "-": np.nan,
        "--": np.nan
    }, inplace=True)

    # Drop NaN rows
    df = df.dropna()

    return df



if __name__ == "__main__":

    df = prepare_dataset()
