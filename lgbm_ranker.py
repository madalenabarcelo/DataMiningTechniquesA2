import numpy as np
import pandas as pd
import os
from lightgbm import LGBMRanker, early_stopping, log_evaluation
from sklearn.model_selection import GroupKFold
from sklearn.utils import shuffle
import copy

class LGBMRankerModel:
    def __init__(self, df:pd.DataFrame,  params:dict) -> None:
        self.ranker = LGBMRanker(**params)
        self.feature_cols = self.get_feature_cols(df)
        # Categorical features should be passed explicitly to the model, these do not include the indentifying columns (srch_id, prop_id)
        self.categorical_features = ["site_id", "visitor_location_country_id", "prop_country_id", "srch_destination_id"]
        self.new_categorical_features = None

    def format_and_train(self, train_df:pd.DataFrame) -> None:
        """Format the training data and train the model."""
        # Format the data
        X_train, y_train, X_val, y_val, groups_size_train, groups_size_val = self.format_data(train_df)

        # Fit the model
        self.fit(X_train, y_train, X_val, y_val, groups_size_train, groups_size_val)
        return X_train, y_train, X_val, y_val, groups_size_train, groups_size_val
    
    def format_data(self, df:pd.DataFrame, group: str = "srch_id") -> tuple[pd.DataFrame, np.array, pd.DataFrame, np.array, np.array]:
        """Format the data for LightGBM."""
        X, y, groups = self.get_X_y_groups(df, group)
        # Split the data into training and validation sets
        X_train, y_train, X_val, y_val, groups_size_train, groups_size_val = self.train_val_split(X, y, groups)

        return X_train, y_train, X_val, y_val, groups_size_train, groups_size_val
    
    def get_X_y_groups(self, df:pd.DataFrame, group: str = "srch_id") -> tuple[pd.DataFrame, np.array, np.array]:
        y = 5 * df["booking_bool"] + 1 * df["click_bool"]
        groups = df[group].to_numpy()
        # Remove the features from the initial small feature selection
        # df = self.remove_initial_features(df)
        # Remove features based on importance from the large feature selection
        df = self.remove_features_on_importance(df, 110)
        X = self.get_X(df)
        return X, y, groups
    
    def remove_initial_features(self, df:pd.DataFrame) -> pd.DataFrame:
        """Remove the features from the initial small feature selection from the DataFrame."""
        features_to_remove = ["comp1_inv_flag", "comp2_inv_flag", "comp6_inv_flag", "comp8_inv_flag", "prop_log_historical_price_zero_flag"]
        self.feature_cols = [col for col in self.feature_cols if col not in features_to_remove]
        return df.drop(columns=features_to_remove)
    
    def remove_features_on_importance(self, df:pd.DataFrame, top_n_features:int = 90) -> pd.DataFrame:
        """Remove features based on importance from the full feature selection from the DataFrame."""
        feature_importance_df = pd.read_parquet("data/feature_importance.parquet")
        # Get the top N features
        top_features = feature_importance_df.head(top_n_features)["feature"].tolist()

        self.feature_cols = top_features

        self.update_categorical_features(top_features)

        return df[self.feature_cols]
    
    def update_categorical_features(self, features:list[str]) -> None:
        self.new_categorical_features = [
            feature for feature in self.categorical_features if feature in features
        ]
        
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
            groups_size_val:np.array, early_stopping_rounds:int = 500, verbose:int = 250) -> LGBMRanker:
        """Fit the model."""
        if not self.new_categorical_features:
            self.new_categorical_features = self.categorical_features
        # Fit the model
        fitted_model = self.ranker.fit(
            X_train, y_train,
            group=groups_size_train,  # counts per srch_id
            eval_set=[(X_val, y_val)],
            eval_group=[groups_size_val],
            callbacks=[early_stopping(early_stopping_rounds), log_evaluation(verbose)],
            categorical_feature=self.new_categorical_features,
        )
        return fitted_model

    def perform_k_fold_fit_and_predict(self, train_df, test_df, n_splits:int = 5, early_stopping_rounds:int = 500, verbose:int = 250):
        all_preds = []
        scores = []
        X_test = self.get_X(test_df)
        X, y, groups = self.get_X_y_groups(train_df)

        if not self.new_categorical_features:
            self.new_categorical_features = self.categorical_features
        i = 0
        for train_idx, val_idx in GroupKFold(n_splits=n_splits).split(X, y, groups):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            groups_train, groups_val = groups[train_idx], groups[val_idx]
            # Get the group sizes
            groups_size_train = np.bincount(groups_train)  # counts per srch_id
            groups_size_val = np.bincount(groups_val)
            # Remove zeros from the group sizes
            groups_size_train = groups_size_train[groups_size_train > 0]
            groups_size_val = groups_size_val[groups_size_val > 0]

            model = copy.deepcopy(self.ranker)

            i += 1
            print(f"Fitting model for fold {i} of {n_splits}")

            fitted_model = model.fit(
                X_train, y_train,
                group=groups_size_train,  # counts per srch_id
                eval_set=[(X_val, y_val)],
                eval_group=[groups_size_val],
                callbacks=[early_stopping(early_stopping_rounds), log_evaluation(verbose)],
                categorical_feature=self.new_categorical_features,
            )
            score = fitted_model.best_score_["valid_0"]["ndcg@5"]
            scores.append(score)
            preds = fitted_model.predict(X_test)
            all_preds.append(preds)
        print(f"Mean score for all {n_splits} folds: {np.mean(scores)}")
        final_test_preds = np.mean(all_preds, axis=0)

        predictions_df = self.add_predictions_to_df(test_df, final_test_preds)
        return predictions_df

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
    
        