# ml_algo/nb.py

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
import numpy as np

def train_and_predict(X_train, y_train, X_test, perform_tuning=False):
    """
    Trains a Multinomial Naive Bayes model with optional hyperparameter tuning.
    
    Args:
        X_train: training features (sparse matrix or array).
        y_train: training labels.
        X_test: test features.
        perform_tuning (bool): If True, tunes the additive smoothing parameter (alpha).
    """
    nb_model = MultinomialNB()

    if perform_tuning:
        print("   - Starting Naive Bayes training with GridSearchCV...")
        
        # alpha is the primary hyperparameter for NB (smoothing)
        param_grid = {
            'alpha': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            'fit_prior': [True, False]
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
        print(f"   - Best alpha found: {grid_search.best_params_['alpha']}")
    else:
        print("   - Training Naive Bayes with default parameters (alpha=1.0)...")
        best_model = nb_model
        best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    
    return y_pred, best_model