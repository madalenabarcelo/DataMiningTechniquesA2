from lgbm_ranker import LGBMRankerModel, LGBMRanker
from data_preparation import DataPreparer
import copy

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

        