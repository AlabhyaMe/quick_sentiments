# MLAlgo/random_forest_model.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import warnings
import numpy as np

def train_and_predict(
    X_train, 
    y_train, 
    X_test, 
    perform_tuning=False, 
    param_grid=None,
    interactive_mode=True,
    random_state=42
):
    """
    Trains RandomForestClassifier model (with optional hyperparameter tuning) and predicts on test data.

    Args:
        X_train: training features (e.g., NumPy array or sparse matrix).
        y_train: training labels (list or NumPy array).
        X_test: test features (e.g., NumPy array or sparse matrix).
        perform_tuning (bool): If True, performs GridSearchCV. If False, trains
                               the model with default parameters. Defaults to True.
        param_grid (dict): Optional custom dictionary for GridSearchCV.
        interactive_mode (bool): If True, prompts for fallback on grid failure.

    Returns:
        y_pred: predicted labels for test set.
        best_model: The best trained RandomForestClassifier model (either from GridSearchCV or simple fit).
    """
    # Base model for training
    rf_model = RandomForestClassifier(n_jobs=-1, random_state=random_state) 

    # A safe defined grid in case none is provided, ensuring the function can run without errors
    default_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'class_weight': [None, 'balanced']
    }

    if perform_tuning:
        if param_grid is None:
            print("   - Starting Random Forest training with DEFAULT PARAMETER, GridSearchCV for hyperparameter tuning...")
            target_grid = default_grid
        else:
            print("   - Starting Random Forest training with CUSTOM PARAMETER, GridSearchCV for hyperparameter tuning...")
            target_grid = param_grid

        grid_search = GridSearchCV(
            estimator=rf_model,
            param_grid=target_grid,
            cv=5,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )

        # Check if grid parameters are valid before fitting
        try:
            grid_search.fit(X_train, y_train)
            
        except ValueError as e:
            # If a custom grid was passed and it failed...
            if param_grid is not None:
                print(f"\n[ERROR] Your custom hyperparameter grid failed: {e}")
                
                if interactive_mode:
                    # The Beginner Path: Ask for permission to fall back
                    user_choice = input("Would you like to fall back to the default parameter grid? (Y/N): ").strip().lower()
                    
                    if user_choice in ['y', 'yes']:
                        print("   - Falling back to default parameters...")
                        grid_search = GridSearchCV(
                            estimator=rf_model,
                            param_grid=default_grid,
                            cv=5,
                            scoring='f1_weighted',
                            n_jobs=-1,
                            verbose=1
                        )
                        grid_search.fit(X_train, y_train)
                    else:
                        print("   - Aborting execution.")
                        raise e 
                else:
                    # The Production Path: Fail fast
                    print("   - Interactive mode is off. Aborting execution.")
                    raise e
            else:
                # If the default grid itself failed (likely a data shape issue), crash it.
                raise e

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
    print("Best model parameters:", best_model.get_params())

    # Return both the predictions and the best model object
    return y_pred, best_model