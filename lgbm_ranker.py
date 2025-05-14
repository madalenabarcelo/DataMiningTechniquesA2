import numpy as np
import pandas as pd
import os
from lightgbm import LGBMRanker, early_stopping, log_evaluation
from sklearn.model_selection import GroupKFold
from sklearn.utils import shuffle

class LGBMRankerModel:
    def __init__(self, df:pd.DataFrame,  **kwargs):
        self.ranker = LGBMRanker(**kwargs)
        self.feature_cols = self.get_feature_cols(df)
        # Categorical features should be passed explicitly to the model, these do not include the indentifying columns (srch_id, prop_id)
        self.categorical_features = ["site_id", "visitor_location_country_id", "prop_country_id", "srch_destination_id"]

    def format_and_train(self, train_df:pd.DataFrame) -> None:
        """Format the training data and train the model."""
        # Format the data
        X_train, y_train, X_val, y_val, groups_size_train, groups_size_val = self.format_data(train_df)

        # Fit the model
        self.fit(X_train, y_train, X_val, y_val, groups_size_train, groups_size_val)
        return X_train, y_train, X_val, y_val, groups_size_train, groups_size_val
    
    def format_data(self, df:pd.DataFrame, group: str = "srch_id") -> tuple[pd.DataFrame, np.array, pd.DataFrame, np.array, np.array]:
        """Format the data for LightGBM."""
        # Remove the features from the initial small feature selection
        df = self.remove_features(df)
        # Remove features based on importance from the large feature selection
        # df = self.remove_features_on_importance(df)
        X = self.get_X(df)
        y = 5 * df["booking_bool"] + 1 * df["click_bool"]
        groups = df[group].to_numpy()

        # Split the data into training and validation sets
        X_train, y_train, X_val, y_val, groups_size_train, groups_size_val = self.train_val_split(X, y, groups)

        return X_train, y_train, X_val, y_val, groups_size_train, groups_size_val
    
    def remove_features(self, df:pd.DataFrame) -> pd.DataFrame:
        features_to_remove = ["comp1_inv_flag", "comp2_inv_flag", "comp6_inv_flag", "comp8_inv_flag", "prop_log_historical_price_zero_flag"]
        self.feature_cols = [col for col in self.feature_cols if col not in features_to_remove]
        return df.drop(columns=features_to_remove)
    
    def remove_features_on_importance(self, df:pd.DataFrame, top_n_features:int = 60) -> pd.DataFrame:
        feature_importance_df = pd.read_parquet("data/feature_importance.parquet")
        # Get the top N features
        top_features = feature_importance_df.head(top_n_features)["feature"].tolist()
        # Remove all but the top N features
        features_to_remove = [col for col in self.feature_cols if col not in top_features]
        self.feature_cols = top_features
        return df.drop(columns=features_to_remove)
    
    def get_X(self, df:pd.DataFrame) -> pd.DataFrame:
        """Get the feature matrix from the DataFrame."""
        X = df[self.feature_cols]
        return X

    def get_feature_cols(self, df:pd.DataFrame) -> list[str]:
        """Get feature columns from the DataFrame. The training set contains multiple columns that are not in the test set."""
        to_exclude = {
            "srch_id",     # group/key for queries
            "prop_id",     # identifier
            "click_bool",  # target
            "booking_bool", # only in train
            "gross_bookings_usd", # only in train
            "position",    # only in train
        }
        feature_cols = [c for c in df.columns if c not in to_exclude]
        return feature_cols
    
    def train_val_split(self, X:pd.DataFrame, y:np.array, groups:np.array) -> tuple[pd.DataFrame, np.array, pd.DataFrame, np.array, np.array]:
        """Split the data into training and validation sets."""
        # Create a group‐aware train/validation split
        gkf = GroupKFold(n_splits=5)
        # Here we take the first fold; you can loop or use more folds
        train_idx, val_idx = next(gkf.split(X, y, groups))
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        groups_train, groups_val = groups[train_idx], groups[val_idx]
        # Get the group sizes
        groups_size_train = np.bincount(groups_train)  # counts per srch_id
        groups_size_val = np.bincount(groups_val)
        # Remove zeros from the group sizes
        groups_size_train = groups_size_train[groups_size_train > 0]
        groups_size_val = groups_size_val[groups_size_val > 0]
        return X_train, y_train, X_val, y_val, groups_size_train, groups_size_val

    def fit(self,X_train:pd.DataFrame, y_train:np.array, X_val:pd.DataFrame, y_val:np.array, groups_size_train:np.array,
            groups_size_val:np.array, early_stopping_rounds:int = 80, verbose:int = 250) -> LGBMRanker:
        """Fit the model."""
        # Check if all the categorical features are in the training set
        for feature in self.categorical_features:
            if feature not in X_train.columns:
                #Remove the feature from the list
                self.categorical_features.remove(feature)

        # Fit the model
        fitted_model = self.ranker.fit(
            X_train, y_train,
            group=groups_size_train,  # counts per srch_id
            eval_set=[(X_val, y_val)],
            eval_group=[groups_size_val],
            callbacks=[early_stopping(early_stopping_rounds), log_evaluation(verbose)],
            categorical_feature=self.categorical_features,
        )
        return fitted_model

    def save_final_results(self, df:pd.DataFrame, file_name:str= "final_predictions") -> None:
        """Save the final results to a CSV file."""
        df = self.get_final_predictions(df).drop(columns=["predictions"])
        # Save the DataFrame as a CSV file
        path  = "data/" + file_name + ".csv"
        if os.path.exists(path):
            os.remove(path)
        df.to_csv(path, index=False)
        print(f"Final results saved to {file_name}")
        print(df)

    def get_final_predictions(self, df:pd.DataFrame) -> pd.DataFrame:
        """Get the property predictions for the test set."""
        df = self.add_predictions(df)
        return df[['srch_id', 'prop_id', 'predictions']]
    
    def add_predictions(self,df:pd.DataFrame) -> pd.DataFrame:
        """Add predictions to a DataFrame."""
        predictions = self.predict(df)
        val_df = self.add_predictions_to_df(df, predictions)
        return val_df

    def predict(self, df:pd.DataFrame) -> np.array:
        """Generate predictions on the test set."""
        X = self.get_X(df)
        return self.ranker.predict(X)
    
    def add_predictions_to_df(self, df:pd.DataFrame, predictions:np.array) -> pd.DataFrame:
        """Add predictions to the DataFrame."""
        df.loc[:, 'predictions'] = predictions
        return df.sort_values(["srch_id", "predictions"], ascending=[True, False])
    
        