# MLAlgo/nn.py

from sklearn.neural_network import MLPClassifier
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
    Trains a Multi-layer Perceptron (MLP) Classifier model (Neural Network)
    (with optional hyperparameter tuning) and predicts on test data.
    """
    # Base model for training, with a max_iter for convergence
    mlp_model = MLPClassifier(random_state=random_state, max_iter=1000)

    # A safe defined grid in case none is provided, ensuring the function can run without errors
    default_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'sgd'],
        'alpha': [0.0001, 0.001, 0.01]
    }

    # --- FIXED: Initialize best_model so it always exists ---
    best_model = None

    if perform_tuning:
        if param_grid is None:
            print("   - Starting MLPClassifier training with DEFAULT PARAMETER, GridSearchCV for hyperparameter tuning...")
            target_grid = default_grid
        else:
            print("   - Starting MLPClassifier training with CUSTOM PARAMETER, GridSearchCV for hyperparameter tuning...")
            target_grid = param_grid

        # Initialize GridSearchCV
        grid_search = GridSearchCV(
            estimator=mlp_model,
            param_grid=target_grid,
            cv=3,  # Using 3-fold cross-validation for speed
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )

        # Check if grid parameters are valid (and fit in RAM) before proceeding
        try:
            grid_search.fit(X_train, y_train)
            
            # --- FIXED: Save the winning model on success ---
            best_model = grid_search.best_estimator_
            
            print("\n   - Best Hyperparameters found:")
            print(grid_search.best_params_)
            print(f"   - Best Cross-Validation Score (F1-weighted): {grid_search.best_score_:.4f}")
            
        except ValueError as e:
            # --- 1. THE INPUT SAFETY NET ---
            if param_grid is not None:
                print(f"\n[ERROR] Your custom hyperparameter grid failed: {e}")
                
                if interactive_mode:
                    user_choice = input("Would you like to fall back to the default parameter grid? (Y/N): ").strip().lower()
                    
                    if user_choice in ['y', 'yes']:
                        print("   - Falling back to default parameters...")
                        grid_search = GridSearchCV(
                            estimator=mlp_model,
                            param_grid=default_grid,
                            cv=3,
                            scoring='f1_weighted',
                            n_jobs=-1,
                            verbose=1
                        )
                        grid_search.fit(X_train, y_train)
                        
                        # --- FIXED: Save the fallback model ---
                        best_model = grid_search.best_estimator_
                        
                        print("\n   - Best Hyperparameters found (Default Grid):")
                        print(grid_search.best_params_)
                    else:
                        print("   - Aborting execution.")
                        raise e 
                else:
                    print("   - Interactive mode is off. Aborting execution.")
                    raise e
            else:
                raise e

        except MemoryError as e:
            # --- 2. THE DATA SAFETY NET ---
            print("\n[CRITICAL ERROR] The cluster ran out of memory while tuning!")
            
            if interactive_mode:
                user_choice = input("Would you like to fall back to training a default model on a 20% random subsample? (Y/N): ").strip().lower()
                
                if user_choice in ['y', 'yes']:
                    print("   - Slicing data matrix and falling back to base MLP Classifier (no grid search)...")
                    
                    # Safely generate random indices
                    sample_size = int(X_train.shape[0] * 0.2)
                    np.random.seed(random_state)
                    indices = np.random.choice(X_train.shape[0], sample_size, replace=False)
                    
                    X_train_small = X_train[indices]
                    y_train_small = y_train[indices]
                    
                    # Abandon grid search, fit the base model
                    best_model = mlp_model
                    best_model.fit(X_train_small, y_train_small)
                    print("   - [SUCCESS] Subsample training complete.")
                else:
                    print("   - Aborting execution.")
                    raise e
            else:
                print("   - Interactive mode is off. Aborting execution.")
                raise e

    else:
        print("   - Training MLPClassifier with default parameters (no hyperparameter tuning)...")
        best_model = mlp_model 
        best_model.fit(X_train, y_train) 
        print("   - Model trained with default parameters.")

    # Make predictions safely using whichever model survived
    y_pred = best_model.predict(X_test)
    y_prob_matrix = best_model.predict_proba(X_test)
    print("\nBest model parameters:", best_model.get_params())

    return y_pred, best_model, y_prob_matrix