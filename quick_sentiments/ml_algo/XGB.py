# ml_algo/XGB.py

from xgboost import XGBClassifier
from xgboost.core import XGBoostError  # <-- NEW: Import XGBoost's internal error class
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import numpy as np

def train_and_predict(
    X_train, 
    y_train, 
    X_test, 
    perform_tuning=False, 
    param_grid=None,
    interactive_mode=False,
    random_state=42
):
    """
    Trains XGBoostClassifier model (with optional hyperparameter tuning) and predicts on test data.
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
    xgb_model = XGBClassifier(
        objective=xgb_objective,
        eval_metric=xgb_eval_metric,
        use_label_encoder=False, 
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

    # --- FIXED: Initialize best_model so it always exists ---
    best_model = None

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
            
            # --- FIXED: Save the winning model on success ---
            best_model = grid_search.best_estimator_
            
            print("\n   - Best Hyperparameters found:")
            print(grid_search.best_params_)
            print(f"   - Best Cross-Validation Score ({scoring_metric}): {grid_search.best_score_:.4f}")
            
        except ValueError as e:
            # --- 1. THE INPUT SAFETY NET ---
            if param_grid is not None:
                print(f"\n[ERROR] Your custom hyperparameter grid failed: {e}")
                
                if interactive_mode:
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
                        
                        # --- FIXED: Save the fallback model ---
                        best_model = grid_search.best_estimator_
                        
                    else:
                        print("   - Aborting execution.")
                        raise e 
                else:
                    print("   - Interactive mode is off. Aborting execution.")
                    raise e
            else:
                raise e

        # --- FIXED: Catch BOTH Python MemoryError and XGBoost's C++ crashes ---
        except (MemoryError, XGBoostError) as e:
            # --- 2. THE DATA SAFETY NET ---
            print("\n[CRITICAL ERROR] The cluster ran out of memory or XGBoost failed while tuning!")
            
            if interactive_mode:
                user_choice = input("Would you like to fall back to training a default model on a 20% random subsample? (Y/N): ").strip().lower()
                
                if user_choice in ['y', 'yes']:
                    print("   - Slicing data matrix and falling back to base XGBoost model (no grid search)...")
                    
                    # Safely generate random indices
                    sample_size = int(X_train.shape[0] * 0.2)
                    np.random.seed(random_state)
                    indices = np.random.choice(X_train.shape[0], sample_size, replace=False)
                    
                    X_train_small = X_train[indices]
                    y_train_small = y_train[indices]
                    
                    # Abandon grid search, fit the base model
                    best_model = xgb_model
                    best_model.fit(X_train_small, y_train_small)
                    print("   - [SUCCESS] Subsample training complete.")
                else:
                    print("   - Aborting execution.")
                    raise e
            else:
                print("   - Interactive mode is off. Aborting execution.")
                raise e

    else:
        print("   - Training XGBoost with default parameters (no hyperparameter tuning)...")
        best_model = xgb_model 
        best_model.fit(X_train, y_train) 
        print("   - Model trained with default parameters.")

    # Make predictions safely using whichever model survived
    y_pred = best_model.predict(X_test)
    y_prob_matrix = best_model.predict_proba(X_test)
    print("\nBest model parameters:", best_model.get_params())

    return y_pred, best_model, y_prob_matrix