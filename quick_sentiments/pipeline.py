# POSITIVELY DO NOT CHANGE

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter
from sklearn.preprocessing import LabelEncoder
import polars as pl
import importlib
import numpy as np
import pandas as pd
from typing import Union

def run_pipeline(
    vectorizer_name: str,
    model_name: str,
    df: Union[pl.DataFrame, pd.DataFrame],
    text_column_name: str,
    sentiment_column_name: str,
    perform_tuning: bool = False,
    random_state: int = 42,
    param_grid: dict = None,
    interactive: bool = True
):
    """
    Orchestrates the full sentiment analysis pipeline: Vectorization -> Training -> Evaluation.

    This function automatically handles data splitting, vectorization (converting text to numbers),
    training the selected machine learning model, and evaluating its performance.

    Args:
        vectorizer_name (str): 
            The method to convert text to numbers. Options:
            - 'tfidf' (Recommended for most cases)
            - 'bow' (Bag of Words - simple counts)
            - 'glove' (Word Embeddings - captures meaning)
            - 'tf' (Term Frequency - raw counts)
            - 'word2vec' (Word2Vec Embeddings - captures meaning)
            - 'hf' (Hugging Face Transformers - state of the art context)
        model_name (str): 
            The machine learning algorithm to use. Options:
            - 'rf' (Random Forest - robust)
            - 'logit' (Logistic Regression - fast baseline)
            - 'xgb' (XGBoost - high performance)
            - 'nb' (Naive Bayes - good for text)
        df (Union[pl.DataFrame, pd.DataFrame]): 
            The input DataFrame containing your data.
        text_column_name (str): 
            The name of the column containing the *cleaned* text.
        sentiment_column_name (str): 
            The name of the column containing the target labels (e.g., 'positive', 'negative').
        perform_tuning (bool, optional): 
            If True, performs Hyperparameter Tuning (GridSearch) to find the best model settings. 
            Warning: This can take a long time! Defaults to False.

    Returns:
        dict: A dictionary containing:
            - 'model_object': The trained model.
            - 'vectorizer_object': The fitted vectorizer.
            - 'accuracy': The accuracy score (float).
            - 'report': A detailed classification report.

    Example:
        >>> results = run_pipeline(
        ...     vectorizer_name="tfidf",
        ...     model_name="rf",
        ...     df=my_data,
        ...     text_column_name="clean_text",
        ...     sentiment_column_name="label"
        ... )
        >>> print(results['accuracy'])
        0.85
    """
    # --- NEW: Mapping for Short and Long Names ---
    VEC_MAP = {
        "bow": "BOW",
        "bag_of_words": "BOW",

        "tfidf": "tfidf",
        "tf_idf": "tfidf",
        "tf-idf": "tfidf",

        "tf": "tf",
        "term_frequency": "tf",

        "glove": "glove_25",
        "glove_25": "glove_25",
        "glove50": "glove_25",
        "glove_50": "glove_50",
        "glove100": "glove_100",
        "glove_100": "glove_100",
        "glove200": "glove_200",
        "glove_200": "glove_200",

        "word2vec": "wv",
        "wv": "wv",

        "hf": "huggingface",
        "huggingface": "huggingface",
        "transformer": "huggingface"
    }

    MODEL_MAP = {
        "rf": "rf",
        "random_forest": "rf",

        "logit": "logit",
        "logistic_regression": "logit",
        "lr": "logit",

        "nb": "nb",
        "naive_bayes": "nb",

        "nn": "nn",
        "mlp": "nn",
        "neural_network": "nn",

        "xgb": "XGB",
        "xgboost": "XGB",
    
        "tf": "tf_model",
        "tensorflow": "tf_model",
        "keras": "tf_model"
    }

    actual_vec_module = VEC_MAP.get(vectorizer_name.lower(), vectorizer_name)
    actual_model_module = MODEL_MAP.get(model_name.lower(), model_name)

    print(f"--- Running Pipeline for {vectorizer_name.replace('_', ' ').title()} + {model_name.replace('_', ' ').title()} ---")

    # Import vectorizer from vect folder
    try:
        vec_module = importlib.import_module(f"quick_sentiments.vect.{actual_vec_module}")
        vectorize_train = getattr(vec_module, "vectorize_train")
        vectorize_test = getattr(vec_module, "vectorize_test")
    except (ImportError, AttributeError) as e:
        print(f"Error loading vectorizer module/function: {e}")
        return None

    # Import ML model from ml_algo folder
    try:
        model_module = importlib.import_module(f"quick_sentiments.ml_algo.{actual_model_module}")
        train_and_predict_function = getattr(model_module, "train_and_predict")
    except (ImportError, AttributeError) as e:
        print(f"Error loading ML model module/function: {e}")
        return None

    """
    Modified to handle both Polars and pandas DataFrames.
    """
    # Convert to Polars if input is pandas
    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)
    elif not isinstance(df, pl.DataFrame):
        raise TypeError(f"Expected Polars or pandas DataFrame, got {type(df)}")
    
      
    # --- NEW: Check for and drop None values in X_text and y_raw ---
    initial_data_len = len(df)
    df = df.drop_nulls(subset=[text_column_name, sentiment_column_name])
    
    
    dropped_rows_count = initial_data_len - len(df)
    if dropped_rows_count > 0:
        print(f"WARNING: Dropped {dropped_rows_count} rows. Rows left: {len(df)}")

    # 2. Extract and Split
    # Only convert to list at the last possible second for the vectorizer
    X_text = df[text_column_name].to_list() 
    
    label_encoder = LabelEncoder()
    # For labels, NumPy is much more memory efficient than a Python list
    y = label_encoder.fit_transform(df[sentiment_column_name].to_numpy())
    
    # Clear the DataFrame from memory if you don't need it anymore
    del df

    # Split data Before vectorization
    print("1. Splitting data into train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=random_state, stratify=y
    )
    del X_text

    # Vectorize the dataset (X)
    print("2. Vectorizing  dataset (X)...")

    #Ensure that we delete the original X_train and X_test after vectorization to free up memory, especially for large datasets.
    try:
        X_train_vectorized, fitted_vectorizer_object, norm = vectorize_train(X_train)
        X_test_vectorized = vectorize_test(X_test, fitted_vectorizer_object, norm)
    finally:
        del X_train, X_test

    # Train + predict
    print("3. Training and predicting...")
    y_pred, trained_model_object,y_prob_matrix = train_and_predict_function(X_train_vectorized, 
                                                              y_train, 
                                                              X_test_vectorized, 
                                                              perform_tuning=perform_tuning,
                                                              random_state=random_state,
                                                              param_grid=param_grid,
                                                              interactive_mode=interactive)
    

    # Evaluate
    print("4. Evaluating model...")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    print("True labels distribution:", Counter(y_test))
    print("Predicted labels distribution:", Counter(y_pred))

    # Return results including all necessary objects for future predictions
    return {
        "model_object": trained_model_object,
        "vectorizer_name": vectorizer_name,
        "vectorizer_object": fitted_vectorizer_object,
        "label_encoder": label_encoder,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob_matrix": y_prob_matrix,
        "norm_object": norm,
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred, output_dict=True, target_names=label_encoder.classes_)
    }