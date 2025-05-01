import pandas as pd
import polars as pl
import os
import matplotlib.pyplot as plt
import seaborn as sns

class DataPreparer:
    def __init__(self):
        pass

    def load_and_preprocess_data(self, old_file_name: str, new_file_name:str) -> None:
        """
        Load and preprocess the data from a CSV file.
        """
        # Load the data
        df = self.load_data(old_file_name)
        
        # Shrink data types
        df = self.shrink_data_types(df)
        
        self.upload_data(df, new_file_name)

    def load_data(self,old_file_name:str, folder:str="data") -> pl.DataFrame:
        path = folder + "/" + old_file_name
        return pl.read_csv(path, low_memory=False, null_values=["NA", "N/A", "null", "NULL", "NaN"])
    
    def shrink_data_types(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Shrink the data types of a Polars DataFrame to save memory.
        """
        return df.select(pl.all().shrink_dtype())
    
    def upload_data(self, df: pl.DataFrame, file_name: str, folder: str = "data") -> None:
        """
        Upload a Polars DataFrame to a CSV file.
        """
        path = folder + "/" + file_name + ".parquet"
        if not os.path.exists(folder):
            os.makedirs(folder)
        df.write_parquet(path)
        print(f"Data uploaded to {path}")