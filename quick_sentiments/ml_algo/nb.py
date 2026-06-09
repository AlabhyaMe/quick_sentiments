# MLAlgo/nb.py

from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import GridSearchCV
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
    Automatically selects GaussianNB for embeddings (Word2Vec) 
    and MultinomialNB for counts (Bag of Words), with safe fallback tuning.
    """
    
    # STEP 1: Check for Negative Values (The "Word2Vec" Detector)
    # Using .min() is safer for sparse matrices than np.min()
    if X_train.min() < 0:
        model_type = "Gaussian"
        print("   - Detected negative values (Embeddings). Switching to Gaussian Naive Bayes.")
        nb_model = GaussianNB()
        
        # Dynamically assign the Gaussian default grid
        default_grid = {
            'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
        }
    else:
        model_type = "Multinomial"
        print("   - Detected all positive values (Counts). Using Multinomial Naive Bayes.")
        nb_model = MultinomialNB()
        
        # Dynamically assign the Multinomial default grid
        default_grid = {
            'alpha': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            'fit_prior': [True, False]
        }

    # --- FIXED: Initialize best_model so it always exists ---
    best_model = None

    # STEP 2: Training with optional Tuning
    if perform_tuning:
        if param_grid is None:
            print(f"   - Starting {model_type} Naive Bayes training with DEFAULT PARAMETER, GridSearchCV for hyperparameter tuning...")
            target_grid = default_grid
        else:
            print(f"   - Starting {model_type} Naive Bayes training with CUSTOM PARAMETER, GridSearchCV for hyperparameter tuning...")
            target_grid = param_grid

        grid_search = GridSearchCV(
            estimator=nb_model,
            param_grid=target_grid,
            cv=5,
            scoring='f1_weighted',
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
                            estimator=nb_model,
                            param_grid=default_grid,
                            cv=5,
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
                    print(f"   - Slicing data matrix and falling back to base {model_type} Naive Bayes (no grid search)...")
                    
                    # Safely generate random indices
                    sample_size = int(X_train.shape[0] * 0.2)
                    np.random.seed(random_state)
                    indices = np.random.choice(X_train.shape[0], sample_size, replace=False)
                    
                    X_train_small = X_train[indices]
                    y_train_small = y_train[indices]
                    
                    # Abandon grid search, fit the base model
                    best_model = nb_model
                    best_model.fit(X_train_small, y_train_small)
                    print("   - [SUCCESS] Subsample training complete.")
                else:
                    print("   - Aborting execution.")
                    raise e
            else:
                print("   - Interactive mode is off. Aborting execution.")
                raise e

    else:
        print(f"   - Training {model_type} Naive Bayes with default parameters (no hyperparameter tuning)...")
        best_model = nb_model
        best_model.fit(X_train, y_train)
        print("   - Model trained with default parameters.")

    # STEP 3: Predict
    y_pred = best_model.predict(X_test)
    y_prob_matrix = best_model.predict_proba(X_test)
    print("Best model parameters:", best_model.get_params())
    
    return y_pred, best_model, y_prob_matrix