from lgbm_ranker import LGBMRankerModel, LGBMRanker, early_stopping, log_evaluation
from data_preparation import DataPreparer
import copy
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import ndcg_score
import optuna


class FeatureTester:
    def __init__(self, model:LGBMRankerModel, categorical_features_threshold: dict[str, int] = {}, impute_null_columns: dict[str,str] = {}) -> None:
        self.model = model
        self.categorical_features_threshold = categorical_features_threshold
        self.impute_null_columns = impute_null_columns

    def test_cardinal_reduction_methods(self, list_of_categorical_features_threshold: list[dict]) -> None:
        for categorical_features_threshold in list_of_categorical_features_threshold:
            # Create a DataPreparer instance
            data_preparer = DataPreparer(impute_null_columns=self.impute_null_columns, categorical_features_threshold=categorical_features_threshold)
            # Load and preprocess the data
            df = data_preparer.load_and_preprocess_data("training_set_VU_DM.csv", upload=False)
            model = copy.deepcopy(self.model)
            X_train, y_train, X_val, y_val, groups_size_train, groups_size_val = model.format_data(df)
            model_opt = model.fit(X_train, y_train, X_val, y_val, groups_size_train, groups_size_val)

            print(f"Model trained successfully with the following categorical features threshold: {categorical_features_threshold}")
            for category in categorical_features_threshold:
                print(f"Number of categories for {category}: {len(df[category].unique())}")
            print(f"Final result gave the following score on the validation set: {model_opt.best_score_}")

    
    def test_na_fill_method(self, impute_null_columns_methods) -> None:
        """
        Test the NA fill method by checking if the model can handle missing values.
        """
        # Create a DataPreparer instance
        for impute_null_columns in impute_null_columns_methods:
            data_preparer_imputed = DataPreparer(impute_null_columns=impute_null_columns, categorical_features_threshold=self.categorical_features_threshold)
            # Load and preprocess the data
            df_imputed = data_preparer_imputed.load_and_preprocess_data("training_set_VU_DM.csv", upload=False)
            model_imputed = copy.deepcopy(self.model)
            X_train, y_train, X_val, y_val, groups_size_train, groups_size_val = model_imputed.format_data(df_imputed)
            model_opt = model_imputed.fit(X_train, y_train, X_val, y_val, groups_size_train, groups_size_val)
            print(f"Model trained successfully with imputing the null values for columns: {impute_null_columns}")
            print(f"Final result gave the following score on the validation set: {model_opt.best_score_}")

    def light_feature_filtering(self, df: pd.DataFrame, target_columns=['srch_id', 'prop_id'], corr_thresh=0.95):
        df = df.copy()
        
        # Drop target columns
        if target_columns is not None:
            target = df[target_columns].copy()
            target_column = target_columns[0]
            df.drop(columns=target_columns, inplace=True)

        # Drop near-constant features (low variance)
        low_variance_cols = df.loc[:, df.nunique() <= 1].columns.tolist()
        print(f"Dropped {len(low_variance_cols)} low-variance columns.")
        print(f"Droppped columns: {low_variance_cols}, with variance: {df[low_variance_cols].var()}")
        df.drop(columns=low_variance_cols, inplace=True)

        # Drop highly correlated columns
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr().abs()

        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        correlated_cols = [column for column in upper.columns if any(upper[column] > corr_thresh)]
        for col in correlated_cols:
            # Show what column the column is correlated to
            correlated_col = upper[col].idxmax()
            print(f"{col} is correlated to {correlated_col} with a correlation of {upper[col].max()}")
        df.drop(columns=correlated_cols, inplace=True)

        # Add target column back
        if target is not None:
            df[target_column] = target

        return df
    
    def objective(self, trial, X_train, y_train, X_val, y_val, groups_size_train, groups_size_val) -> float:
        # Define search space
        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "ndcg_eval_at": [5],
            "n_estimators": 1000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 10, 100),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True)
        }

        scores = []

        model = LGBMRanker(**params)
        model.fit( X_train, y_train,
            group=groups_size_train,  # counts per srch_id
            eval_set=[(X_val, y_val)],
            eval_group=[groups_size_val],
            callbacks=[early_stopping(80), log_evaluation(100)],
            categorical_feature=self.model.categorical_features)
        
        # Use NDCG@5 as evaluation
        ndcg = model.best_score_['valid_0']['ndcg@5']
        scores.append(ndcg)

        return np.mean(scores)

    def run_light_hpo(self, X_train, y_train, X_val, y_val, groups_size_train, groups_size_val, n_trials=30):
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: self.objective(trial, X_train, y_train, X_val, y_val, groups_size_train, groups_size_val), n_trials=n_trials)
        print("Best trial:")
        print(study.best_trial)
        return study.best_params