from lgbm_ranker import LGBMRankerModel, LGBMRanker
from data_preparation import DataPreparer
import copy

class FeatureTester:
    def __init__(self, model:LGBMRankerModel, impute_null_columns: dict[str, str] = {}):
        self.model = model
        self.impute_null_columns = impute_null_columns
    
    def test_na_fill_method(self) -> tuple[LGBMRanker, LGBMRanker]:
        """
        Test the NA fill method by checking if the model can handle missing values.
        """
        # Create a DataPreparer instance
        data_preparer_imputed = DataPreparer(impute_null_columns=self.impute_null_columns)
        # Load and preprocess the data
        df_imputed = data_preparer_imputed.load_and_preprocess_data("training_set_VU_DM.csv", upload=False)
        model_imputed = copy.deepcopy(self.model)
        X_train, y_train, X_val, y_val, groups_size_train, groups_size_val = model_imputed.format_data(df_imputed)
        model_opt = model_imputed.fit(X_train, y_train, X_val, y_val, groups_size_train, groups_size_val)

        data_preparer_not_imputed = DataPreparer()
        # Load and preprocess the data without imputing
        df_not_imputed = data_preparer_not_imputed.load_and_preprocess_data("training_set_VU_DM.csv", upload=False)
        model_not_imputed = copy.deepcopy(self.model)
        X_train_not_imputed, y_train_not_imputed, X_val_not_imputed, y_val_not_imputed, groups_size_train_not_imputed, groups_size_val_not_imputed = model_not_imputed.format_data(df_not_imputed)
        model_opt_not_imputed = model_not_imputed.fit(X_train_not_imputed, y_train_not_imputed, X_val_not_imputed, y_val_not_imputed, groups_size_train_not_imputed, groups_size_val_not_imputed)
        # Compare the results

        print(f"Model trained successfully with imputing the null values for columns: {self.impute_null_columns}")
        print(f"Final result gave the following score on the validation set: {model_opt.best_score_}")

        print("Model trained successfully without imputing the null values for columns")
        print(f"Final result gave the following score on the validation set: {model_opt_not_imputed.best_score_}")
        return model_opt, model_opt_not_imputed