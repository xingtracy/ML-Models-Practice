import os
import pandas as pd
import kagglehub
import shutil
import pandas as pd

KAGGLE_DATASET = "mdsultanulislamovi/student-stress-monitoring-datasets"
CSV_FILENAME = "Stress_Dataset.csv"  # CSV file name inside dataset
LOCAL_DIR = os.path.join(os.getcwd(), "student-stress-monitoring-datasets")
PARQUET_FILE = os.path.join(LOCAL_DIR, "Student_Stress_Monitoring_Datasets.parquet")

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
    df = simplify_stress_column(df)

    if df.empty:
        raise ValueError("No rows left after cleaning — check filtering rules.")

    df.to_parquet(PARQUET_FILE, index=False)
    print("Saved cleaned dataset to Parquet.")
    return df


def simplify_stress_column(df):
    """
    Cleans the stress column so each value is mapped to one of:
    'Eustress', 'Distress', 'No Stress'.
    """
    last_col=df.columns[-1]
    
    def map_stress(sentence: str) -> str:
        s = sentence.lower()  # normalize case
        if "distress" in s:
            return "Distress"
        elif "eustress" in s:
            return "Eustress"
        elif "no stress" in s:
            return "No Stress"
        else:
            # fallback if unclear -> you can choose default
            return "No Stress"

    df[last_col] = df[last_col].apply(map_stress)
    
    return df




if __name__ == "__main__":

    df = prepare_dataset()
