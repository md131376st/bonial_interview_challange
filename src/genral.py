import os
import pandas as pd
from typing import List

def save_dataframes_to_pickle(dataframes: List[pd.DataFrame], file_names: List[str], folder_path: str) -> None:

    os.makedirs(folder_path, exist_ok=True)

    for df, file_name in zip(dataframes, file_names):
        full_path = os.path.join(folder_path, file_name)  # Full file path
        df.to_pickle(full_path)  # Save as pickle
        print(f"File saved successfully: {full_path}")


def restore_dataframes_from_pickle(file_names: List[str], folder_path: str) -> List[pd.DataFrame]:
    dataframes = []

    for file_name in file_names:
        full_path = os.path.join(folder_path, file_name)  # Full file path
        try:
            df = pd.read_pickle(full_path)  # Load pickle
            dataframes.append(df)
            print(f"File restored successfully: {full_path}")
        except FileNotFoundError:
            print(f"Error: File not found - {full_path}")
        except Exception as e:
            print(f"Error restoring {full_path}: {e}")

    return dataframes
