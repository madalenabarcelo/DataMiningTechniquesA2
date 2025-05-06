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
        
        # Shrink data types before processing to make it faster
        df = self.shrink_data_types(df)

        # Handle date and time columns and add features related to it
        df = self.handle_date_time(df)

        # Shrink data types again after processing to save memory
        df = self.shrink_data_types(df)
        
        # Save the processed data as parquet
        self.upload_data(df, new_file_name)

    def load_data(self,old_file_name:str, folder:str="data") -> pl.DataFrame:
        path = folder + "/" + old_file_name
        return pl.read_csv(path, low_memory=False, null_values=["NA", "N/A", "null", "NULL", "NaN"])
    
    def shrink_data_types(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Shrink the data types of a Polars DataFrame to save memory.
        """
        return df.select(pl.all().shrink_dtype())

    def handle_date_time(self, df: pl.DataFrame) -> pl.DataFrame:
        """Handle date and time columns in the DataFrame. And add features related to it. """
        # Convert date columns to datetime
        df = df.with_columns(pl.col("date_time").str.strptime(format="%Y-%m-%d %H:%M:%S", dtype=pl.Datetime))
        
        df = df.with_columns([
            pl.col("date_time").dt.hour().alias("hour"),
            pl.col("date_time").dt.weekday().alias("weekday"),
            pl.col("date_time").dt.month().alias("month"),
            (pl.col("date_time").dt.hour().is_between(6, 12)).cast(pl.Int8).alias("is_morning"),
            (pl.col("date_time").dt.hour().is_between(12, 18)).cast(pl.Int8).alias("is_afternoon"),
            (pl.col("date_time").dt.hour().is_between(18, 24)).cast(pl.Int8).alias("is_evening"),
            (pl.col("date_time").dt.hour().is_between(0, 6)).cast(pl.Int8).alias("is_night"),
            (pl.col("date_time").dt.month().is_between(6, 8)).cast(pl.Int8).alias("is_summer"),
            (pl.col("date_time").dt.month().is_between(9, 11)).cast(pl.Int8).alias("is_autumn"),
            (pl.col("date_time").dt.month().is_between(12, 2)).cast(pl.Int8).alias("is_winter"),
            (pl.col("date_time").dt.month().is_between(3, 5)).cast(pl.Int8).alias("is_spring"),
            (pl.col("date_time").dt.weekday().is_between(5, 6)).cast(pl.Int8).alias("is_weekend"),
        ])
        
        return df.drop("date_time")

    def upload_data(self, df: pl.DataFrame, file_name: str, folder: str = "data") -> None:
        """
        Upload a Polars DataFrame to a CSV file.
        """
        path = folder + "/" + file_name + ".parquet"
        if not os.path.exists(folder):
            os.makedirs(folder)
        # If it already exists, remove it
        if os.path.exists(path):
            os.remove(path)
        # Save the DataFrame as a parquet file
        df.write_parquet(path)
        print(f"Data uploaded to {path}")