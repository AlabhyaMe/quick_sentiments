from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import GridSearchCV
import numpy as np

def train_and_predict(X_train, y_train, X_test, perform_tuning=False):
    """
    Automatically selects GaussianNB for embeddings (Word2Vec) 
    and MultinomialNB for counts (Bag of Words).
    """
    
    # STEP 1: Check for Negative Values (The "Word2Vec" Detector)
    # If the data has negative numbers, it's likely an embedding -> Use GaussianNB
    # If the data is all positive, it's likely counts -> Use MultinomialNB
    if np.min(X_train) < 0:
        model_type = "Gaussian"
        print("   - Detected negative values (Embeddings). Switching to Gaussian Naive Bayes.")
        nb_model = GaussianNB()
    else:
        model_type = "Multinomial"
        print("   - Detected all positive values (Counts). Using Multinomial Naive Bayes.")
        nb_model = MultinomialNB()

    # STEP 2: Training with optional Tuning
    if perform_tuning:
        print(f"   - Starting {model_type} Naive Bayes training with GridSearchCV...")
        
        # Define different grids based on the model type
        if model_type == "Multinomial":
            param_grid = {
                'alpha': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
                'fit_prior': [True, False]
            }
        else:
            # GaussianNB doesn't have 'alpha'. It uses 'var_smoothing'.
            param_grid = {
                'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
            }
        
        grid_search = GridSearchCV(
            estimator=nb_model,
            param_grid=param_grid,
            cv=5,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print(f"   - Best params found: {grid_search.best_params_}")
        
    else:
        print(f"   - Training {model_type} Naive Bayes with default parameters...")
        best_model = nb_model
        best_model.fit(X_train, y_train)

    # STEP 3: Predict
    y_pred = best_model.predict(X_test)
    
    return y_pred, best_model