# ml_algo/tf_model.py

import numpy as np
import scipy.sparse as sp
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
    interactive_mode=False,
    balance_classes=False,
    random_state=42
):
    """
    Trains a TensorFlow Neural Network seamlessly within the scikit-learn pipeline.
    """
    tf.random.set_seed(random_state)

    if balance_classes:
        print("   - Balancing classes via Random Oversampling for TensorFlow...")
        classes, counts = np.unique(y_train, return_counts=True)
        max_count = np.max(counts)
        
        X_resampled_list = []
        y_resampled_list = []
        
        for c in classes:
            c_indices = np.where(y_train == c)[0]
            # Randomly sample with replacement to match the majority class count
            resampled_indices = np.random.choice(c_indices, max_count, replace=True)
            X_resampled_list.append(X_train[resampled_indices])
            y_resampled_list.append(y_train[resampled_indices])
            
        # Safely combine data whether it's a sparse matrix or dense array
        if sp.issparse(X_train):
            X_train = sp.vstack(X_resampled_list)
        else:
            X_train = np.vstack(X_resampled_list)
            
        y_train = np.concatenate(y_resampled_list)
        
        # Shuffle the newly ordered data to prevent training bias
        np.random.seed(random_state)
        shuffle_idx = np.random.permutation(len(y_train))
        X_train = X_train[shuffle_idx]
        y_train = y_train[shuffle_idx]
    
    # Handle Sparse Data from Bag-of-Words / TF-IDF
    if sp.issparse(X_train):
        print("   - Converting sparse text matrix to dense array for TensorFlow...")
        X_train = X_train.toarray()
        X_test = X_test.toarray()

    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    print(f"   - Building TF Model: Input Features = {input_dim}, Output Classes = {num_classes}")
    
    keras_estimator = KerasClassifier(
        model=build_keras_model,
        model__input_dim=input_dim,
        model__num_classes=num_classes,
        epochs=10,
        batch_size=32,
        verbose=0 
    )

    default_grid = {
        'model__hidden_units': [64, 128],
        'model__dropout_rate': [0.2, 0.4],
        'batch_size': [32, 64]
    }

    # --- FIXED: Initialize best_model so it always exists ---
    best_model = None

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
            
            # --- FIXED: Save the winning model on success ---
            best_model = grid_search.best_estimator_
            
            print("\n   - Best Hyperparameters found:")
            print(grid_search.best_params_)
            print(f"   - Best Cross-Validation Score (Accuracy): {grid_search.best_score_:.4f}")
            
        except ValueError as e:
            # --- 1. THE INPUT SAFETY NET ---
            if param_grid is not None:
                print(f"\n[ERROR] Your custom hyperparameter grid failed: {e}")
                
                if interactive_mode:
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

        # --- FIXED: Catch BOTH Python MemoryError and TensorFlow ResourceExhaustedError ---
        except (MemoryError, tf.errors.ResourceExhaustedError) as e:
            # --- 2. THE DATA SAFETY NET ---
            print("\n[CRITICAL ERROR] The cluster ran out of memory while tuning!")
            
            if interactive_mode:
                user_choice = input("Would you like to fall back to training a default model on a 20% random subsample? (Y/N): ").strip().lower()
                
                if user_choice in ['y', 'yes']:
                    print("   - Slicing data matrix and falling back to base TF model (no grid search)...")
                    
                    sample_size = int(X_train.shape[0] * 0.2)
                    np.random.seed(random_state)
                    indices = np.random.choice(X_train.shape[0], sample_size, replace=False)
                    
                    X_train_small = X_train[indices]
                    y_train_small = y_train[indices]
                    
                    best_model = keras_estimator
                    # We add verbose=1 here so the user sees the safety net actually working
                    best_model.fit(X_train_small, y_train_small, epochs=10, batch_size=32, verbose=1)
                    print("   - [SUCCESS] Subsample training complete.")
                else:
                    print("   - Aborting execution.")
                    raise e
            else:
                print("   - Interactive mode is off. Aborting execution.")
                raise e
        
    else:
        print("   - Training TensorFlow model with default parameters...")
        best_model = keras_estimator
        best_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

    print("   - Generating predictions...")
    y_pred = best_model.predict(X_test)
    y_prob_matrix = best_model.predict_proba(X_test)

    return y_pred, best_model, y_prob_matrix
