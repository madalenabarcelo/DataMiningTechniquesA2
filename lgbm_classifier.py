import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

class LGBMClassifierModel:
    def __init__(self, df:pd.DataFrame,  **kwargs):
        self.model = LGBMClassifier(**kwargs)
        self.feature_cols = self.get_feature_cols(df)
        # Categorical features should be passed explicitly to the model, these do not include the indentifying columns (srch_id, prop_id)
        self.categorical_features = ["site_id", "visitor_location_country_id", "prop_country_id", "srch_destination_id"]

    def format_and_train(self, train_df:pd.DataFrame) -> None:
        """Format the training data and train the model."""
        # Format the data
        X_train, y_train, X_val, y_val, val_idx = self.format_data(train_df)

        # Fit the model
        self.fit(X_train, y_train, X_val, y_val)
        return X_train, y_train, X_val, y_val, val_idx

    def format_data(self, df:pd.DataFrame, target:str = "click_bool", group: str = "srch_id") -> tuple[pd.DataFrame, np.array, pd.DataFrame, np.array, np.array]:
        """Format the data for LightGBM."""
        X = self.get_X(df)
        y = df[target].to_numpy()
        groups = df[group].to_numpy()

        # Split the data into training and validation sets
        X_train, y_train, X_val, y_val, val_idx = self.train_val_split(X, y, groups)

        return X_train, y_train, X_val, y_val, val_idx
    
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
        return X_train, y_train, X_val, y_val, val_idx
    
    def fit(self, X_train:pd.DataFrame, y_train:np.array, X_val:pd.DataFrame, y_val:np.array, eval_metric: str = "auc", 
            early_stopping_rounds:int = 50, verbose:int = 20) -> None:
        """Fit the model."""
        # Fit the model
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=eval_metric,
            callbacks=[early_stopping(early_stopping_rounds), log_evaluation(verbose)],
            categorical_feature=self.categorical_features,
        )

    def evaluate_validation(self, X_val:pd.DataFrame, y_val:np.array, train_df: pd.DataFrame, val_idx) -> float:
        """Evaluate the model."""
        # Get the ROC AUC score
        roc_auc = self.get_roc_auc(X_val, y_val)

        val_df:pd.DataFrame = train_df.iloc[val_idx]

        # Compute NDCG score
        mean_ndcg = self.get_ndcg_score(val_df)
        return roc_auc, mean_ndcg
    
    def get_ndcg_score(self, df:pd.DataFrame, k:int = 5) -> float:
        """Get the NDCG score."""
        df = self.add_predictions(df)
        df = df[["srch_id", "click_bool", "score"]]
        df = df.sort_values(["srch_id", "score"], ascending=[True, False])

        # Compute NDCG@k
        mean_ndcg = self.mean_ndcg_at_k(df, k=k)
        print(f"Validation NDCG@{k}:", mean_ndcg)
        return mean_ndcg

    def mean_ndcg_at_k(self, df:pd.DataFrame, k:int=5):
        """Compute mean NDCG@k across all srch_ids"""
        ndcgs = []
        for srch_id, group in df.groupby("srch_id"):
            # Relevance is typically click_bool or booking_bool (or a combined score)
            relevances = group["click_bool"].tolist()  # or use combined click + 5*booking
            ndcgs.append(self.ndcg_at_k(relevances, k))
        return np.mean(ndcgs)
    
    def ndcg_at_k(self, relevances: list[float], k: int) -> float:
        """Normalized DCG at rank k"""
        ideal = sorted(relevances, reverse=True)
        ideal_dcg = self.dcg(ideal, k)
        return self.dcg(relevances, k) / ideal_dcg if ideal_dcg > 0 else 0.0
    
    def dcg(self, relevances: list[float], k: int) -> float:
        """Discounted Cumulative Gain at rank k"""
        relevances = np.array(relevances)[:k]
        return np.sum((2 ** relevances - 1) / np.log2(np.arange(2, len(relevances) + 2)))
    
    def get_roc_auc(self, X_val: pd.DataFrame, y_val:np.array) -> float:
        """Get the ROC AUC score."""
        y_val_pred = self.model.predict_proba(X_val)[:,1]
        roc_auc = roc_auc_score(y_val, y_val_pred)
        print("Validation AUC:", roc_auc)
        return roc_auc

    def get_final_predictions(self, df:pd.DataFrame) -> np.array:
        """Get the property predictions for the test set."""
        df = self.add_predictions(df)
        return df[['srch_id', 'prop_id', 'score']]
    
    def add_predictions(self,df:pd.DataFrame) -> pd.DataFrame:
        """Add predictions to a DataFrame."""
        predictions = self.predict(df)
        val_df = self.add_predictions_to_df(df, predictions)
        return val_df

    def predict(self, df:pd.DataFrame) -> np.array:
        """Generate predictions on the test set."""
        X = self.get_X(df)
        return self.model.predict_proba(X)[:,1]
    
    def add_predictions_to_df(self, df:pd.DataFrame, predictions:np.array) -> pd.DataFrame:
        """Add predictions to the DataFrame."""
        df.loc[:, 'score'] = predictions
        return df.sort_values(["srch_id", "score"], ascending=[True, False])
    
        