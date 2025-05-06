# 1. Imports
import polars as pl
import numpy as np
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

class LGBMClassifierModel:
    def __init__(self, **kwargs):
        self.model = LGBMClassifier(**kwargs)

    def format_and_train(self, train_df:pl.DataFrame) -> None:
        """Format the training data and train the model."""
        # Format the data
        X_train, y_train, X_val, y_val = self.format_data(train_df)

        # Fit the model
        self.fit(X_train, y_train, X_val, y_val)

    def format_data(self, df:pl.DataFrame) -> tuple[np.array, np.array, np.array, np.array]:
        """Format the data for LightGBM."""
        feature_cols = self.get_feature_cols(df)

        # Extract feature matrix, target array, and query‐groups
        X = df.select(feature_cols).to_numpy()
        y = df["click_bool"].to_numpy()
        groups = df["srch_id"].to_numpy()

        # Split the data into training and validation sets
        X_train, y_train, X_val, y_val = self.train_val_split(X, y, groups)
        return X_train, y_train, X_val, y_val

    def get_feature_cols(self, df:pl.DataFrame) -> list[str]:
        """Get feature columns from the DataFrame. The training set contains multiple columns that are not in the test set."""
        to_exclude = {
            "srch_id",     # group/key for queries
            "prop_id",     # identifier
            "click_bool",  # target
            "booking_bool",# only used for NDCG grading later
            "gross_booking_usd",
            "position",    # only in train
        }
        feature_cols = [c for c in df.columns if c not in to_exclude]
        return feature_cols
    
    def train_val_split(self, X:np.array, y:np.array, groups:np.array) -> tuple[np.array, np.array, np.array, np.array]:
        """Split the data into training and validation sets."""
        # Create a group‐aware train/validation split
        gkf = GroupKFold(n_splits=5)
        # Here we take the first fold; you can loop or use more folds
        train_idx, val_idx = next(gkf.split(X, y, groups))
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        return X_train, y_train, X_val, y_val
    
    def fit(self, X_train:np.array, y_train:np.array, X_val:np.array, y_val:np.array, eval_metric: str = "auc", 
            early_stopping_rounds:int = 50, verbose:int = 20) -> None:
        """Fit the model."""
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=eval_metric,
            callbacks=[early_stopping(early_stopping_rounds), log_evaluation(verbose)],
        )

    def evaluate(self, X_val:np.array, y_val:np.array) -> float:
        """Evaluate the model."""
        y_val_pred = self.model.predict_proba(X_val)[:,1]
        roc_auc = roc_auc_score(y_val, y_val_pred)
        print("Validation AUC:", roc_auc)
        return roc_auc
    
    def get_predictions(self, test_df:pl.DataFrame) -> np.array:
        """Get the property predictions for the test set."""
        predictions = self.predict(test_df)
        prop_predictions = self.get_prop_predictions(test_df, predictions)
        return prop_predictions

    def predict(self, test_df:pl.DataFrame) -> np.array:
        """Generate predictions on the test set."""
        feature_cols = self.get_feature_cols(test_df)
        X_test = test_df.select(feature_cols).to_numpy()
        return self.model.predict_proba(X_test)[:,1]
    
    def get_prop_predictions(self, test_df:pl.DataFrame, predictions:np.array) -> pl.DataFrame:
        """Get property predictions."""
        test_df = test_df.with_columns(score = pl.Series(predictions))
        return test_df.sort(["srch_id", "score"], reversed=[False, True]).select(["srch_id", "prop_id"])
        