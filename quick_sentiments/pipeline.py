# POSITIVELY DO NOT CHANGE (This comment is for the user, I am making necessary fixes based on our conversation)

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
    perform_tuning: bool = False
):
    """
    Runs the full pipeline:
      - vectorize
      - train model
      - evaluate

    Args:
        vectorizer_name (str): Name of the vectorizer (e.g., 'tfidf', 'word_embedding').
        model_name (str): Name of the ML model (e.g., 'logistic_regression', 'random_forest').
        df (pl.DataFrame): Your Polars DataFrame containing the text and sentiment columns.
        text_column_name (str): The name of the column in `df` that contains the processed text.
        sentiment_column_name (str): The name of the column in `df` that contains the sentiment labels.

    Returns:
        dict: A dictionary containing the trained model, fitted vectorizer, label encoder, and evaluation results.
    """
    print(f"--- Running Pipeline for {vectorizer_name.replace('_', ' ').title()} + {model_name.replace('_', ' ').title()} ---")

    # Import vectorizer from vect folder
    try:
        vec_module = importlib.import_module(f"quick_sentiments.vect.{vectorizer_name}")
        vectorize_train = getattr(vec_module, "vectorize_train")
        vectorize_test = getattr(vec_module, "vectorize_test")
    except (ImportError, AttributeError) as e:
        print(f"Error loading vectorizer module/function: {e}")
        return None

    # Import ML model from ml_algo folder
    try:
        model_module = importlib.import_module(f"quick_sentiments.ml_algo.{model_name}")
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
    
    # Polars DataFrame handling
    X_text = df[text_column_name].to_list()
    y_raw = df[sentiment_column_name].to_list()
    
    # --- NEW: Check for and drop None values in X_text and y_raw ---
    initial_data_len = len(X_text)
    
    # Filter out pairs where either X_text element or y_raw element is None
    # Use zip to iterate over both lists simultaneously and filter
    filtered_data = [(x, y_val) for x, y_val in zip(X_text, y_raw) if x is not None and y_val is not None]
    
    # Unzip the filtered data back into X_text and y_raw
    if filtered_data: # Check if filtered_data is not empty to avoid unpacking error
        X_text, y_raw = zip(*filtered_data)
        X_text = list(X_text) # Convert back to list
        y_raw = list(y_raw)   # Convert back to list
    else:
        # Handle case where all data might be None
        print("WARNING: All data rows contained missing values after initial extraction. Cannot proceed with training.")
        return None

    dropped_rows_count = initial_data_len - len(X_text)
    if dropped_rows_count > 0:
        print(f"WARNING: Dropped {dropped_rows_count} rows due to missing values (None) in '{text_column_name}' or '{sentiment_column_name}' columns. Original rows: {initial_data_len}, Rows after dropping: {len(X_text)}")
    else:
        print("No missing values (None) found in text or sentiment columns. Proceeding with all rows.")
    # ------------------------------------------------------------------

    # Label Encoding for y_raw
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    print(f"Labels encoded: Original -> {label_encoder.classes_}, Encoded -> {np.unique(y)}")

    # Split data Before vectorization
    print("1. Splitting data into train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )

    # Vectorize the dataset (X)
    print("2. Vectorizing  dataset (X)...")
    X_train_vectorized, fitted_vectorizer_object,norm = vectorize_train(X_train)
    X_test_vectorized = vectorize_test(X_test, fitted_vectorizer_object,norm)

    # Train + predict
    print("3. Training and predicting...")
    y_pred, trained_model_object = train_and_predict_function(X_train_vectorized, y_train, X_test_vectorized, perform_tuning=perform_tuning)

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
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred, output_dict=True, target_names=label_encoder.classes_)
    }