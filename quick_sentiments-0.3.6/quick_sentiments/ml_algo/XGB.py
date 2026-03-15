from xgboost import XGBClassifier # Import XGBClassifier
from sklearn.model_selection import GridSearchCV # Import GridSearchCV
from sklearn.metrics import classification_report # For evaluation metrics
import numpy as np # For type hinting

def train_and_predict(X_train, y_train, X_test, perform_tuning: bool = True, random_state = 42): # <--- Added perform_tuning flag and random_state
    """
    Trains XGBoostClassifier model (with optional hyperparameter tuning) and predicts on test data.

    Args:
        X_train: training features (e.g., NumPy array or sparse matrix).
        y_train: training labels (numerical, e.g., 0, 1, 2...).
        X_test: test features (e.g., NumPy array or sparse matrix).
        perform_tuning (bool): If True, performs GridSearchCV. If False, trains
                               the model with default parameters. Defaults to True.

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

    # Base XGBClassifier model (used for both tuning and simple fit)
    # verbosity=0 to suppress excessive output from XGBoost itself during GridSearchCV
    xgb_model = XGBClassifier(
        objective=xgb_objective,
        eval_metric=xgb_eval_metric,
        use_label_encoder=False, # Suppress warning for newer versions
        random_state=42,
        num_class=num_classes if num_classes > 2 else None,
        verbosity=0 # Suppress XGBoost internal verbosity during grid search
    )


    if perform_tuning:
        print("   - Performing GridSearchCV for hyperparameter tuning...")

       
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
        
        
        }
        print("   - Using default parameter grid for tuning:", param_grid)

        grid_search = GridSearchCV(
            estimator=xgb_model, # Use the base xgb_model here
            param_grid=param_grid,
            cv=5,
            scoring=scoring_metric,
            n_jobs=-1,
            verbose=1 # Print progress messages from GridSearchCV
        )

        # Fit GridSearchCV to the training data
        grid_search.fit(X_train, y_train)

        # Get the best model found by GridSearchCV
        best_model = grid_search.best_estimator_

        print("\n   - Best Hyperparameters found:")
        print(grid_search.best_params_)
        print(f"   - Best Cross-Validation Score ({scoring_metric}): {grid_search.best_score_:.4f}")
    else:
        
        print("   - Training XGBoost with default parameters (no hyperparameter tuning)...")
        best_model = xgb_model # Use the base model directly
        best_model.fit(X_train, y_train) # Fit it on X_train, y_train
        print("   - Model trained with default parameters.")

    # Make predictions on the test set using the best model (tuned or default)
    y_pred = best_model.predict(X_test)

    # Return both the predictions and the best model object
    return y_pred, best_model