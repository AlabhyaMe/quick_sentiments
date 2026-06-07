# ml_algo/XGB.py

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
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
    Trains XGBoostClassifier model (with optional hyperparameter tuning) and predicts on test data.

    Args:
        X_train: training features (e.g., NumPy array or sparse matrix).
        y_train: training labels (numerical, e.g., 0, 1, 2...).
        X_test: test features (e.g., NumPy array or sparse matrix).
        perform_tuning (bool): If True, performs GridSearchCV. If False, trains
                               the model with default parameters. Defaults to False.
        param_grid (dict): Optional custom dictionary for GridSearchCV.
        interactive_mode (bool): If True, prompts for fallback on grid failure.

    Returns:
        y_pred: predicted labels for test set.
        best_model: The best trained XGBoostClassifier model (either from GridSearchCV or simple fit).
    """
    print("   - Starting XGBoost training...")

    # Determine objective and eval_metric based on number of unique classes
    num_classes = len(np.unique(y_train))
    
    if num_classes == 2:
        xgb_objective = 'binary:logistic'
        xgb_eval_metric = 'logloss'
        scoring_metric = 'f1_weighted'
    else:
        xgb_objective = 'multi:softmax'
        xgb_eval_metric = 'mlogloss'
        scoring_metric = 'f1_weighted' # Or 'accuracy'

    # Base XGBClassifier model
    # verbosity=0 to suppress excessive output from XGBoost itself during GridSearchCV
    xgb_model = XGBClassifier(
        objective=xgb_objective,
        eval_metric=xgb_eval_metric,
        use_label_encoder=False, # Suppress warning for newer versions
        random_state=random_state,
        num_class=num_classes if num_classes > 2 else None,
        verbosity=0 
    )

    # A safe defined grid in case none is provided, ensuring the function can run without errors
    default_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    if perform_tuning:
        if param_grid is None:
            print("   - Starting XGBoost training with DEFAULT PARAMETER, GridSearchCV for hyperparameter tuning...")
            target_grid = default_grid
        else:
            print("   - Starting XGBoost training with CUSTOM PARAMETER, GridSearchCV for hyperparameter tuning...")
            target_grid = param_grid

        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=target_grid,
            cv=5,
            scoring=scoring_metric,
            n_jobs=-1,
            verbose=1 
        )

        # The Safety Net
        try:
            grid_search.fit(X_train, y_train)
            
        except Exception as e:  # <-- Catch Exception because XGBoost throws XGBoostError
            # If a custom grid was passed and it failed...
            if param_grid is not None:
                print(f"\n[ERROR] Your custom hyperparameter grid failed: {e}")
                
                if interactive_mode:
                    # The Beginner Path: Ask for permission to fall back
                    user_choice = input("Would you like to fall back to the default parameter grid? (Y/N): ").strip().lower()
                    
                    if user_choice in ['y', 'yes']:
                        print("   - Falling back to default parameters...")
                        grid_search = GridSearchCV(
                            estimator=xgb_model,
                            param_grid=default_grid,
                            cv=5,
                            scoring=scoring_metric,
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
                # If the default grid itself failed, crash it.
                raise e

        best_model = grid_search.best_estimator_

        print("\n   - Best Hyperparameters found:")
        print(grid_search.best_params_)
        print(f"   - Best Cross-Validation Score ({scoring_metric}): {grid_search.best_score_:.4f}")
        
    else:
        print("   - Training XGBoost with default parameters (no hyperparameter tuning)...")
        best_model = xgb_model 
        best_model.fit(X_train, y_train) 
        print("   - Model trained with default parameters.")

    # Make predictions on the test set
    y_pred = best_model.predict(X_test)
    print("Best model parameters:", best_model.get_params())

    return y_pred, best_model