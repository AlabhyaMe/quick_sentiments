from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import numpy as np 

def train_and_predict(X_train, y_train, X_test, perform_tuning=False, random_state=42):
    """
    Trains a Multi-layer Perceptron (MLP) Classifier model (Neural Network)
    (with optional hyperparameter tuning) and predicts on test data.

    Args:
        X_train: training features (e.g., NumPy array or sparse matrix).
        y_train: training labels (list or NumPy array).
        X_test: test features (e.g., NumPy array or sparse matrix).
        perform_tuning (bool): If True, performs GridSearchCV. If False, trains
                               the model with default parameters. Defaults to False.

    Returns:
        y_pred: predicted labels for test set.
        best_model: The best trained MLPClassifier model (either from GridSearchCV or simple fit).
    """
    # Base model for training, with a max_iter for convergence
    mlp_model = MLPClassifier(random_state=random_state, max_iter=1000)

    if perform_tuning:
        print("   - Starting MLPClassifier training with GridSearchCV for hyperparameter tuning...")

        # Define the parameter grid to search
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'sgd'],
            'alpha': [0.0001, 0.001, 0.01]
        }
        print("   - Using default parameter grid for tuning:", param_grid)

        # Initialize GridSearchCV
        grid_search = GridSearchCV(
            estimator=mlp_model,
            param_grid=param_grid,
            cv=3,  # Using 3-fold cross-validation for speed
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print("\n   - Best Hyperparameters found:")
        print(grid_search.best_params_)
        print(f"   - Best Cross-Validation Score (F1-weighted): {grid_search.best_score_:.4f}")
    else:
        print("   - Training MLPClassifier with default parameters (no hyperparameter tuning)...")
        best_model = mlp_model # Use the base model directly
        best_model.fit(X_train, y_train) # Fit it on X_train, y_train
        print("   - Model trained with default parameters.")

    y_pred = best_model.predict(X_test)
    print("Best model parameters:", best_model.get_params())

    return y_pred, best_model

