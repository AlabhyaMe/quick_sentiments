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
                            estimator=nb_model,
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
        print(f"   - Training {model_type} Naive Bayes with default parameters (no hyperparameter tuning)...")
        best_model = nb_model
        best_model.fit(X_train, y_train)
        print("   - Model trained with default parameters.")

    # STEP 3: Predict
    y_pred = best_model.predict(X_test)
    print("Best model parameters:", best_model.get_params())
    
    return y_pred, best_model