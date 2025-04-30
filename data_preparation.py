import pandas as pd
import polars as pl
import os
import matplotlib.pyplot as plt
import seaborn as sns

class DataPreparer:
    def __init__(self):
        pass

    def load_data(self,path) -> pl.DataFrame:
        return pl.read_csv(path, low_memory=False)
    
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