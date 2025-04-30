import pandas as pd
import polars as pl
import os
import matplotlib.pyplot as plt
import seaborn as sns

class DataExplorer:
    def __init__(self):
        pass

    def load_data(self,path) -> pd.DataFrame:
        return pd.read_csv(path, low_memory=False)
        
    def get_expedia_data_info(self,df: pd.DataFrame):
        """
        Get detailed information about a wide-format Expedia dataset.
        """
        print(f"Number of observations (rows): {df.shape[0]}")
        print(f"Number of features (columns): {df.shape[1]}")
        
        # Number of unique values per column
        print("\nUnique values per column:")
        n_unique = df.nunique().sort_values(ascending=False)
        print(n_unique)
        
        # Number of null values per column
        print("\nMissing (null) values per column:")
        null_values = df.isnull().sum().sort_values(ascending=False)
        print(null_values[null_values > 0])

        # Summary statistics for numeric columns
        print("\nSummary statistics for numeric columns:")
        numeric_stats = df.describe(include=[float, int]).transpose()[['mean', 'min', 'max']]
        print(numeric_stats)

        # Summary statistics for object/categorical columns
        print("\nMost common values for categorical columns:")
        cat_cols = df.select_dtypes(include='object').columns
        for col in cat_cols:
            print(f"{col} top values:")
            print(df[col].value_counts(dropna=False).head(5))
            print()

        return {
            "n_unique": n_unique,
            "null_values": null_values,
            "numeric_stats": numeric_stats,
        }

    
    def get_expedia_data_plots(self, df: pd.DataFrame, max_bins=50, max_columns=20):
        """
        Generate plots for key data characteristics in a wide-format Expedia dataset.
        """

        # Plot missing values
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if not missing.empty:
            plt.figure(figsize=(12, 6))
            sns.barplot(x=missing.index, y=missing.values, palette="viridis")
            plt.title("Missing Values per Column")
            plt.ylabel("Count")
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.show()

        # Plot cardinality of each column (number of unique values)
        unique_counts = df.nunique().sort_values(ascending=False)
        plt.figure(figsize=(12, 6))
        sns.barplot(x=unique_counts.index[:max_columns], y=unique_counts.values[:max_columns], palette="crest")
        plt.title(f"Top {max_columns} Columns by Unique Value Count")
        plt.xticks(rotation=90)
        plt.ylabel("Unique Values")
        plt.tight_layout()
        plt.show()

        # Plot distribution for numeric columns
        num_cols = df.select_dtypes(include=[int, float]).columns
        for col in num_cols[:max_columns]:  # Limit to first N columns for speed
            plt.figure(figsize=(8, 4))
            sns.histplot(df[col].dropna(), bins=max_bins, kde=True)
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.tight_layout()
            plt.show()


    def plot_missing_values(self, df):
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        plt.figure(figsize=(12, 6))
        sns.barplot(x=missing.index, y=missing.values)
        plt.xticks(rotation=90)
        plt.title("Missing Values per Column")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

    def plot_numeric_distributions(self,df, columns, max_bins=50):
        for col in columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                plt.figure(figsize=(8, 4))
                sns.histplot(df[col].dropna(), bins=max_bins, kde=False)
                plt.title(f"Distribution of {col}")
                plt.xlabel(col)
                plt.tight_layout()
                plt.show()

    def booking_click_rates(self, df):
        if 'booking_bool' in df.columns and 'click_bool' in df.columns:
            print("Booking Rate:", df['booking_bool'].mean())
            print("Click Rate:", df['click_bool'].mean())
            sns.countplot(x='booking_bool', data=df)
            plt.title("Booking Bool Distribution")
            plt.show()
            sns.countplot(x='click_bool', data=df)
            plt.title("Click Bool Distribution")
            plt.show()
        else:
            print("Columns 'booking_bool' or 'click_bool' not found in dataframe.")

    def correlate_features_with_target(self, df, target='booking_bool'):
        if target not in df.columns:
            print(f"Target column '{target}' not found.")
            return
        corr = df.corr(numeric_only=True)[target].sort_values(ascending=False)
        print(f"Top correlations with {target}:")
        print(corr.head(10))
        print("\nLowest correlations:")
        print(corr.tail(10))
