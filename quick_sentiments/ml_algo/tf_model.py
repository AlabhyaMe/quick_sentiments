# ml_algo/tf_model.py

import numpy as np
from scipy.sparse import issparse
from sklearn.model_selection import GridSearchCV
import warnings

# 1. Gracefully handle the Optional Dependencies
try:
    import tensorflow as tf
    from scikeras.wrappers import KerasClassifier
except ImportError:
    raise ImportError(
        "TensorFlow or SciKeras is missing. "
        "To use the deep learning models, please install the optional dependencies:\n"
        "pip install quick_sentiments[dl]"
    )

def build_keras_model(input_dim, num_classes, hidden_units=128, dropout_rate=0.3):
    """Builds the core TensorFlow architecture for SciKeras to use."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(hidden_units, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(int(hidden_units/2), activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

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
    Trains a TensorFlow Neural Network seamlessly within the scikit-learn pipeline.
    """
    tf.random.set_seed(random_state)
    
    # Handle Sparse Data from Bag-of-Words / TF-IDF
    if issparse(X_train):
        print("   - Converting sparse text matrix to dense array for TensorFlow...")
        X_train = X_train.toarray()
        X_test = X_test.toarray()

    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    print(f"   - Building TF Model: Input Features = {input_dim}, Output Classes = {num_classes}")
    
    # Wrap the Keras model so it acts exactly like your MLPClassifier
    keras_estimator = KerasClassifier(
        model=build_keras_model,
        model__input_dim=input_dim,
        model__num_classes=num_classes,
        epochs=10,
        batch_size=32,
        verbose=0 
    )

    # A safe defined grid in case none is provided, ensuring the function can run without errors
    default_grid = {
        'model__hidden_units': [64, 128],
        'model__dropout_rate': [0.2, 0.4],
        'batch_size': [32, 64]
    }

    if perform_tuning:
        if param_grid is None:
            print("   - Starting TensorFlow training with DEFAULT PARAMETER, GridSearchCV for hyperparameter tuning...")
            target_grid = default_grid
        else:
            print("   - Starting TensorFlow training with CUSTOM PARAMETER, GridSearchCV for hyperparameter tuning...")
            target_grid = param_grid
        
        grid_search = GridSearchCV(
            estimator=keras_estimator,
            param_grid=target_grid,
            cv=3, 
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        # The Safety Net
        try:
            grid_search.fit(X_train, y_train)
            
        except Exception as e:  # <-- Broadened to catch various TF/Keras crash types
            # If a custom grid was passed and it failed...
            if param_grid is not None:
                print(f"\n[ERROR] Your custom hyperparameter grid failed: {e}")
                
                if interactive_mode:
                    # The Beginner Path: Ask for permission to fall back
                    user_choice = input("Would you like to fall back to the default parameter grid? (Y/N): ").strip().lower()
                    
                    if user_choice in ['y', 'yes']:
                        print("   - Falling back to default parameters...")
                        grid_search = GridSearchCV(
                            estimator=keras_estimator,
                            param_grid=default_grid,
                            cv=3,
                            scoring='accuracy',
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
                # If the default grid itself failed (likely an OOM or shape issue), crash it.
                raise e

        best_model = grid_search.best_estimator_
        print("\n   - Best Hyperparameters found:")
        print(grid_search.best_params_)
        print(f"   - Best Cross-Validation Score (Accuracy): {grid_search.best_score_:.4f}")
        
    else:
        print("   - Training TensorFlow model with default parameters...")
        best_model = keras_estimator
        # verbose=1 shows the progress bar when not tuning
        best_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

    print("   - Generating predictions...")
    y_pred = best_model.predict(X_test)

    return y_pred, best_model