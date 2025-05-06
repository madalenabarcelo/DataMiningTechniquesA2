import pandas as pd
import polars as pl
import os
import matplotlib.pyplot as plt
import seaborn as sns

class DataPreparer:
    def __init__(self):
        # These are the columns where nan has some meaning
        self.flag_columns = (
            ["visitor_hist_starrating", "visitor_hist_adr_usd", "prop_review_score", 
            "srch_query_affinity_score", "orig_destination_distance"]
            + [f"comp{i}_rate" for i in range(1, 9)]
            + [f"comp{i}_inv" for i in range(1, 9)]
            + [f"comp{i}_rate_percent_diff" for i in range(1, 9)]
        )
        # In these columns we will impute the null values with either the mean, median or whatever (to be decided)
        self.impute_null_columns = [] 
        
        # These are the columns where zero has some meaning
        self.flag_zero_columns = ["prop_review_score", "prop_log_historical_price"]

    def load_and_preprocess_data(self, old_file_name: str, new_file_name:str) -> None:
        """
        Load and preprocess the data from a CSV file.
        """
        # Load the data
        df = self.load_data(old_file_name)

        # Handle date and time columns and add features related to it
        df = self.handle_date_time(df)

        # Cast all non-numeric columns to numeric
        df = self.cast_to_numeric(df)
        
        # Shrink data types before processing to make it faster
        df = self.shrink_data_types(df)

        # Add flag columns for null and zeros to the DataFrame
        df = self.handle_missing_values(df)

        # Add features to the DataFrame
        df = self.add_features(df)

        # Shrink data types again after processing to save memory
        df = self.shrink_data_types(df)
        
        # Save the processed data as parquet
        self.upload_data(df, new_file_name)

    def load_data(self,old_file_name:str, folder:str="data") -> pl.DataFrame:
        path = folder + "/" + old_file_name
        return pl.read_csv(path, low_memory=False, null_values=["NA", "N/A", "null", "NULL", "NaN"])
    
    def cast_to_numeric(self, df: pl.DataFrame) -> pl.DataFrame:
        """Cast all non-numeric columns to numeric. First try to cast to int8, int16, then to float32."""
        for col in df.columns:
            if (df[col].dtype == pl.Object) or (df[col].dtype == pl.Utf8):
                try:
                    df = df.with_columns(pl.col(col).cast(pl.Int8))
                except Exception:
                    try:
                        df = df.with_columns(pl.col(col).cast(pl.Int16))
                    except Exception:
                        df = df.with_columns(pl.col(col).cast(pl.Float32))
        return df
    
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
            (pl.col("date_time").dt.month().is_between(3, 5)).cast(pl.Int8).alias("is_spring"),
            (pl.col("date_time").dt.month().is_between(6, 8)).cast(pl.Int8).alias("is_summer"),
            (pl.col("date_time").dt.month().is_between(9, 11)).cast(pl.Int8).alias("is_autumn"),
            (pl.col("date_time").dt.month().is_in([1,2,12])).cast(pl.Int8).alias("is_winter"),
            (pl.col("date_time").dt.weekday().is_between(5, 6)).cast(pl.Int8).alias("is_weekend"),
        ])
        
        return df.drop("date_time")
    
    def handle_missing_values(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Add flag columns to the DataFrame. The flag columns are created by checking if the values in the specified columns are null.
        For selected columns, the null values are imputed with the mean of the column grouped by "srch_id".
        The zero values in the specified columns are also flagged and replaced with NaN.
        """
        for col in self.flag_columns:
            # Create a flag column for each column in the list
            df = df.with_columns((pl.col(col).is_null()).cast(pl.Int8).alias(f"{col}_flag"))

        if len(self.impute_null_columns) > 0:
            # For the columns where we want to impute the null values, we use the mean grouped by "srch_id" and replace
            for col in self.impute_null_columns:
                # Do with mean for now
                df_mean = df.group_by("srch_id").agg([
                    pl.col(col).mean().alias(f"{col}_mean")
                ])
            # Merge the mean values back to the original DataFrame
            df = df.join(df_mean, on="srch_id", how="left")

            for col in self.impute_null_columns:
                # Impute the null values with the mean values
                df = df.with_columns(
                    pl.when(pl.col(col).is_null())
                    .then(pl.col(f"{col}_mean"))
                    .otherwise(pl.col(col))
                    .alias(col)
                ).drop(f"{col}_mean")

        for col in self.flag_zero_columns:
            # Create a flag column for each column in the list
            df = df.with_columns((pl.col(col) == 0).cast(pl.Int8).alias(f"{col}_zero_flag"))
            # From what I see the zero value could be interpreted by the model as a value rather than a missing data flag
            # Therefore it makes sense to replace the zero value with nan
            df = df.with_columns(pl.when(pl.col(col) == 0).then(None).otherwise(pl.col(col)).alias(col))
            # Also for both current columns imputing it does not make sense so keep it nan

        ### Consider what to do with nan in original columns (see notes)
        ### Consider what to do with prop_starrating (see notes)
        return df
    
    def add_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Add features to the DataFrame.
        """
        df = df.with_columns([
            pl.col("srch_id").count().over("srch_id").alias("result_count"),
            pl.when(pl.col("visitor_location_country_id") == pl.col("prop_country_id"))
            .then(1).otherwise(0).alias("prop_in_visitor_country"),
        ])

        
        return df

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