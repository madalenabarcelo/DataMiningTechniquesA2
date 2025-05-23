import pandas as pd
import polars as pl
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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

    def plot_missing_values_both(self, train_df, test_df):
        # Calculate missing counts
        missing_train = train_df.isnull().sum()
        missing_train = missing_train[missing_train > 0].sort_values(ascending=False)
        missing_test = test_df.isnull().sum()
        missing_test = missing_test.reindex(missing_train.index).fillna(0)  # Align index, fill NaNs with 0

        # Combine into one DataFrame for plotting
        missing_df = pd.DataFrame({
            'column': missing_train.index.tolist() * 2,
            'missing_count': list(missing_train.values) + list(missing_test.values),
            'dataset': ['train'] * len(missing_train) + ['test'] * len(missing_test)
        })

        plt.figure(figsize=(12, 6))
        sns.barplot(data=missing_df, x='column', y='missing_count', hue='dataset', palette='viridis')
        plt.title("Missing Values per Column (Train vs Test)")
        plt.ylabel("Count")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

    
    def get_expedia_data_plots(self, train_df: pd.DataFrame, test_df: pd.DataFrame, max_bins=50, max_columns=20):
        """
        Generate comparison plots for train and test Expedia datasets.
        """

        # Add source label
        train_df = train_df.copy()
        test_df = test_df.copy()
        train_df["dataset"] = "train"
        test_df["dataset"] = "test"

        df_combined = pd.concat([train_df, test_df], ignore_index=True)

        self.plot_missing_values_both(train_df, test_df)

        

        # Specify the variables of interest
        cat_vars = ['site_id', 'visitor_location_country_id', 'prop_country_id', 'srch_destination_id']

        # Compute unique counts
        unique_counts_train = train_df[cat_vars].nunique()
        unique_counts_test = test_df[cat_vars].nunique()

        # Create DataFrame for plotting
        unique_df = pd.DataFrame({
            'column': unique_counts_train.index,
            'train': unique_counts_train.values,
            'test': unique_counts_test.reindex(unique_counts_train.index).values
        })

        # Melt for seaborn
        unique_df_melted = unique_df.melt(id_vars="column", var_name="dataset", value_name="unique_count")

        # Plot
        plt.figure(figsize=(8, 5))
        sns.barplot(data=unique_df_melted, x="column", y="unique_count", hue="dataset", palette="Set2")
        plt.title("Unique Value Count: Selected Categorical Features (Train vs Test)")
        plt.ylabel("Unique Values")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()

        # --- Numeric Distributions ---
        num_cols = train_df.select_dtypes(include=[int, float]).columns.intersection(test_df.columns)

        for col in num_cols[:max_columns]:
            combined_data = df_combined[col].dropna()

            # Compute consistent bins for train & test
            bin_edges = np.histogram_bin_edges(combined_data, bins=max_bins)

            plt.figure(figsize=(8, 4))
            sns.histplot(data=df_combined, x=col, hue="dataset", bins=bin_edges,
                        kde=True, element="step", stat="density", common_norm=False)
            plt.title(f"Distribution of {col} (Train vs Test)")
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
