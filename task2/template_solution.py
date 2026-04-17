# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # Required to use IterativeImputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, DotProduct, RBF, Matern, RationalQuadratic
from sklearn.model_selection import cross_val_score

def load_data():
    """
    This function loads the training and test data, preprocesses it, removes the NaN values and interpolates the missing
    data using imputation

    Parameters
    ----------
    Returns
    ----------
    X_train: matrix of floats, training input with features
    y_train: array of floats, training output with labels
    X_test: matrix of floats: dim = (100, ?), test input with features
    """
    # Load training data
    train_df = pd.read_csv("train.csv")
    
    print("Training data:")
    print("Shape:", train_df.shape)
    print(train_df.head(2))
    print('\n')
    
    # Load test data
    test_df = pd.read_csv("test.csv")

    print("Test data:")
    print(test_df.shape)
    print(test_df.head(2))
    print('\n')

    # Handle missing TARGET values in training data
    # Drop rows where the target 'price_CHF' is missing
    train_df = train_df.dropna(subset=['price_CHF'])
    
    y_train = train_df['price_CHF'].values
    X_train = train_df.drop(columns=['price_CHF'])
    X_test = test_df.copy()

    assert (X_train.shape[1] == X_test.shape[1]) and (X_train.shape[0] == y_train.shape[0]) and (X_test.shape[0] == 100), "Invalid data shape"
    return X_train, y_train, X_test


class Model(object):
    def __init__(self):
        super().__init__()
        self._best_pipeline = None
        self._best_score = -np.inf
        self._best_combo_name = ""

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Fit the model by evaluating all combinations of imputers and kernels using cross-validation."""
        
        # Define feature groups
        categorical_features = ['season'] if 'season' in X_train.columns else []
        numeric_features = [col for col in X_train.columns if col not in categorical_features]
        
        # Define the combinations we want to test
        imputation_strategies = {
            'Iterative': IterativeImputer(random_state=42)
        }

        kernels = {
            'RationalQuadratic * Matern': RationalQuadratic() * ConstantKernel() * Matern() * ConstantKernel(),
        }

        print("Evaluating all combinations using Cross-Validation (R2 Score)...\n")

        # Iterate over every combination
        for imp_name, imputer in imputation_strategies.items():
            for kern_name, kernel in kernels.items():
                
                # Create the preprocessing steps for numeric features
                numeric_transformer = Pipeline(steps=[
                    ('imputer', imputer),
                    ('scaler', StandardScaler())
                ])

                # Create the preprocessing steps for categorical features
                categorical_transformer = None
                if categorical_features:
                    categorical_transformer = Pipeline(steps=[
                        ('imputer', SimpleImputer(strategy='most_frequent')),
                        ('onehot', OneHotEncoder(handle_unknown='ignore'))
                    ])

                # Combine them using ColumnTransformer
                transformers = [('num', numeric_transformer, numeric_features)]
                if categorical_transformer is not None:
                    transformers.append(('cat', categorical_transformer, categorical_features))
                    
                preprocessor = ColumnTransformer(transformers=transformers)

                # Create the full pipeline with the regressor
                model = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('regressor', GaussianProcessRegressor(kernel=kernel, random_state=42, normalize_y=True))
                ])

                # Evaluate using 5-fold cross-validation
                # Use R2 scoring as requested in the task description
                scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                mean_score = scores.mean()
                
                combo_name = f"Imputer: {imp_name} | Kernel: {kern_name}"
                print(f"{combo_name} --> Mean R2: {mean_score:.4f}")

                # Keep track of the best performing combination
                if mean_score > self._best_score:
                    self._best_score = mean_score
                    self._best_pipeline = model
                    self._best_combo_name = combo_name

        # Train the best combination on the full training data
        print("\n" + "="*50)
        print(f"Best Combination: {self._best_combo_name} with Validation R2: {self._best_score:.4f}")
        print("="*50 + "\n")
        
        self._best_pipeline.fit(X_train, y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the best fitted model to make predictions."""
        y_pred = self._best_pipeline.predict(X_test)
        assert y_pred.shape == (X_test.shape[0],), "Invalid data shape"
        return y_pred

# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    X_train, y_train, X_test = load_data()
    model = Model()
    # Use this function to fit the model
    model.fit(X_train=X_train, y_train=y_train)
    # Use this function for inference
    y_pred = model.predict(X_test)
    # Save results in the required format
    dt = pd.DataFrame(y_pred) 
    dt.columns = ['price_CHF']
    dt.to_csv('results.csv', index=False)
    print("\nResults file successfully generated!")

