# MLAlgo/random_forest_model.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV # Import GridSearchCV
from sklearn.metrics import classification_report # For evaluation metrics
import numpy as np # For type hinting

def train_and_predict(X_train, y_train, X_test, perform_tuning=False, random_state=42):
    
    """
    Trains RandomForestClassifier model (with optional hyperparameter tuning) and predicts on test data.

    Args:
        X_train: training features (e.g., NumPy array or sparse matrix).
        y_train: training labels (list or NumPy array).
        X_test: test features (e.g., NumPy array or sparse matrix).
        perform_tuning (bool): If True, performs GridSearchCV. If False, trains
                               the model with default parameters. Defaults to True.

    Returns:
        y_pred: predicted labels for test set.
        best_model: The best trained RandomForestClassifier model (either from GridSearchCV or simple fit).
    """
    rf_model = RandomForestClassifier(n_jobs=-1,random_state=random_state) # Base model for training

    if perform_tuning:
        print("   - Starting Random Forest training with GridSearchCV for hyperparameter tuning...")

        # Define the parameter grid to search for RandomForestClassifier (default grid)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'class_weight': [None, 'balanced']
        }
        print("   - Using default parameter grid for tuning:", param_grid)

        grid_search = GridSearchCV(
            estimator=rf_model, # Use the base rf_model here
            param_grid=param_grid,
            cv=5,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )

        # Fit GridSearchCV to the training data
        grid_search.fit(X_train, y_train)

        # Get the best model found by GridSearchCV
        best_model = grid_search.best_estimator_

        print("\n   - Best Hyperparameters found:")
        print(grid_search.best_params_)
        print(f"   - Best Cross-Validation Score (F1-weighted): {grid_search.best_score_:.4f}")
    else:
        print("   - Training Random Forest with default parameters (no hyperparameter tuning)...")
        best_model = rf_model # Use the base model directly
        best_model.fit(X_train, y_train) # Fit it on X_train, y_train
        print("   - Model trained with default parameters.")

    # Make predictions on the test set using the best model (tuned or default)
    y_pred = best_model.predict(X_test)

    # Return both the predictions and the best model object
    return y_pred, best_model