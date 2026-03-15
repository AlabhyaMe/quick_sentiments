# MLAlgo/logistic_regression_model.py

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV # Import GridSearchCV
from sklearn.metrics import classification_report # For evaluation metrics
import numpy as np # For type hinting

def train_and_predict(X_train, y_train, X_test, perform_tuning=False, random_state=42):
    """
        Trains Logistic Regression model (with optional hyperparameter tuning) and predicts on test data.

    Args:
        X_train: training features (e.g., NumPy array or sparse matrix).
        y_train: training labels (list or NumPy array).
        X_test: test features (e.g., NumPy array or sparse matrix).
        perform_tuning (bool): If True, performs GridSearchCV. If False, trains
                               the model with default parameters. Defaults to True.

    Returns:
        y_pred: predicted labels for test set.
        best_model: The best trained LogisticRegression model (either from GridSearchCV or simple fit).
    """
    lr_model = LogisticRegression(random_state=random_state) # Base model for training

    if perform_tuning:
        print("   - Starting Logistic Regression training with GridSearchCV for hyperparameter tuning...")

        # Define the parameter grid to search (default grid, as no custom grid is passed here)
        param_grid = {
            'solver': ['liblinear', 'lbfgs'],
            'C': [0.1, 1.0, 10.0],
            'class_weight': [None, 'balanced'],
            'max_iter': [500, 1000]
        }
        print("   - Using default parameter grid for tuning:", param_grid)

        grid_search = GridSearchCV(
            estimator=lr_model,
            param_grid=param_grid,
            cv=5,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print("\n   - Best Hyperparameters found:")
        print(grid_search.best_params_)
        print(f"   - Best Cross-Validation Score (F1-weighted): {grid_search.best_score_:.4f}")
    else:
        print("   - Training Logistic Regression with default parameters (no hyperparameter tuning)...")
        best_model = lr_model # Use the base model directly
        best_model.fit(X_train, y_train) # Fit it on X_train, y_train
        print("   - Model trained with default parameters.")

    y_pred = best_model.predict(X_test)
    print("Best model parameters:", best_model.get_params())

    return y_pred, best_model