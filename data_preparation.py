import pandas as pd
import polars as pl
import numpy as np
import os
from typing import Optional

class DataPreparer:
    def __init__(self, impute_null_columns: Optional[list] = None, 
                 categorical_features_threshold: Optional[dict] = None):
        # These are the columns where nan has some meaning
        self.flag_columns = (
            ["visitor_hist_starrating", "visitor_hist_adr_usd", "prop_review_score", 
            "srch_query_affinity_score", "orig_destination_distance"]
            + [f"comp{i}_rate" for i in range(1, 9)]
            + [f"comp{i}_inv" for i in range(1, 9)]
            + [f"comp{i}_rate_percent_diff" for i in range(1, 9)]
        )
        # These are the columns where zero has some meaning
        self.flag_zero_columns = ["prop_review_score", "prop_log_historical_price"]

        if impute_null_columns is not None:
            self.impute_null_columns = impute_null_columns
        else:
            # In these columns we will impute the null values with either the mean, median or whatever (to be decided)
            self.impute_null_columns = {}

        self.categorical_features_threshold = categorical_features_threshold if categorical_features_threshold is not None else {}
        # self.categorical_features_threshold = {}
        

    def load_and_preprocess_data(self, old_file_name: str, new_file_name:Optional[str] = None, upload:bool = True) -> Optional[pd.DataFrame]:
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

        # Reduce cardinality of the categorical columns
        df = self.reduce_cardinality(df)

        # Shrink data types again after processing to save memory
        df = self.convert_to_pandas(df)
        
        if upload:
            # Save the processed data as parquet
            self.upload_data(df, new_file_name)
        else:
            return df

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
        # Before handling missing values, we set the np.inf in price_usd to nan
        df = df.with_columns(pl.when(pl.col("price_usd") == np.inf).then(None).otherwise(pl.col("price_usd")).cast(pl.Float32).alias("price_usd"))

        for col in self.flag_columns:
            # Create a flag column for each column in the list
            df = df.with_columns((pl.col(col).is_null()).cast(pl.Int8).alias(f"{col}_flag"))

        for col in self.flag_zero_columns:
            if col in self.flag_columns:
                # If the column is also in the flag_columns, we should not confuscate the null and zero values when imputing
                # So we only create a flag column
                df = df.with_columns((pl.col(col) == 0).cast(pl.Int8).alias(f"{col}_zero_flag"))
            else:
                # From what I see the zero value could be interpreted by the model as a value rather than a missing data flag
                # Therefore it makes sense to replace the zero value with nan
                # Also for both current columns imputing it does not make sense so keep it nan
                df = df.with_columns(pl.when(pl.col(col) == 0).then(None).otherwise(pl.col(col)).alias(col))
                # Create a flag column for each column in the list
                df = df.with_columns((pl.col(col).is_null()).cast(pl.Int8).alias(f"{col}_zero_flag"))

        if len(self.impute_null_columns) > 0:
            # For the columns where we want to impute the null values, we use the mean grouped by "srch_id" and replace
            for col in self.impute_null_columns:
                if self.impute_null_columns[col] == "mean":
                    df = df.with_columns(pl.col(col).cast(pl.Float32))
                    df = df.with_columns(pl.when(pl.col(col).is_null()).then(pl.col(col).mean()).otherwise(pl.col(col)).alias(col))
                elif self.impute_null_columns[col] == "median":
                    df = df.with_columns(pl.when(pl.col(col).is_null()).then(pl.col(col).median()).otherwise(pl.col(col)).alias(col))
        
        # Now we set the zero values to nan for the columns in both the flag and flag_zero columns
        for col in (set(self.flag_zero_columns) - set(self.flag_columns)):
            df = df.with_columns(pl.when(pl.col(col) == 0).then(None).otherwise(pl.col(col)).alias(col))

        ### Consider what to do with prop_starrating (see notes)
        return df
    
    def add_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Add features to the DataFrame.
        """
        df = df.with_columns([
            pl.col("srch_id").count().over("srch_id").alias("result_count"),
            pl.when(pl.col("visitor_location_country_id") == pl.col("prop_country_id"))
            .then(1).otherwise(0).cast(pl.Int8).alias("prop_in_visitor_country"),
        ])

        # Add averages per prop_id (mainly just to reduce noise I think in pricing)
        for col in ["price_usd", "prop_log_historical_price", "prop_location_score1", "prop_location_score2"]:
            df = df.with_columns([
                pl.col(col).mean().over("prop_id").cast(pl.Float32).alias(f"{col}_mean_over_prop_id"),
                pl.col(col).std().over("prop_id").cast(pl.Float32).alias(f"{col}_std_over_prop_id"),
            ])

        for col in ["prop_starrating", "prop_review_score"]:
            df = df.with_columns([
                pl.col(col).median().over("prop_id").cast(pl.Float32).alias(f"{col}_median_over_prop_id"),
                pl.col(col).std().over("prop_id").cast(pl.Float32).alias(f"{col}_std_over_prop_id"),
            ])
        return df
    
    def reduce_cardinality(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Reduce the cardinality of the categorical columns in the DataFrame. Due to limitations im just gonna hard code this for now
        """
        if len(self.categorical_features_threshold) > 0:
            for col, threshold in self.categorical_features_threshold.items():
                original_type = df[col].dtype
                value_counts_df = df[col].value_counts()
                # Filter to values with count < threshold
                rare_values = value_counts_df.filter(pl.col("count") < threshold).select(col).to_series().to_list()
                # Replace those values with 0
                replacements = {val: 0 for val in rare_values}
                df = df.with_columns(pl.col(col).replace(replacements).cast(original_type).alias(col))

        return df

    def convert_to_pandas(self, df: pl.DataFrame) -> pd.DataFrame:
        """Convert the Polars DataFrame to a Pandas DataFrame and shrink the data types as much as possible."""
        # Convert to Pandas DataFrame
        df_pd = df.to_pandas()

        # Shrink the float types of the DataFrame
        # This first batch was acting difficult so im just doing it manually (due to nan values in int columns)
        for col in [f"comp{i}_rate_percent_diff" for i in range(1, 9)]:
            df_pd[col] = df_pd[col].astype("Int32")
        # These are then the other comp columns that were annoying
        for col in df_pd.select_dtypes(include=["float64"]).columns:
            df_pd[col] = df_pd[col].astype("Int8")

        # With pandas we can use the float16 type
        # For some reason the price_usd columns return inf values so we'll just keep them as float32
        df_pd = df_pd.astype({col: np.float16 for col in df_pd.select_dtypes(include=["float32"]).columns if col.startswith("price_usd")})

        return df_pd


    def upload_data(self, df: pd.DataFrame, file_name: str, folder: str = "data") -> None:
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
        df.to_parquet(path)
        print(f"Data uploaded to {path}")