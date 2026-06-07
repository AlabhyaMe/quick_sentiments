from typing import Union, Dict, Any
import polars as pl
import pandas as pd
import numpy as np
from scipy.sparse import issparse
import warnings

def make_predictions(
        new_data: Union[pl.DataFrame, pd.DataFrame],
        text_column_name: str,
        trained_results: Dict[str, Any],
        prediction_column_name: str = "predictions",
        return_confidence: bool = True) -> pl.DataFrame:
    """
    Generates predictions for new data using the results dictionary from run_pipeline.

    Args:
        new_data (Union[pl.DataFrame, pd.DataFrame]): 
            The input DataFrame containing the text to predict on.
        text_column_name (str): 
            The name of the column in `new_data` that contains the *preprocessed* text.
        trained_results (Dict[str, Any]): 
            The dictionary returned by `run_pipeline`. Must contain:
            - "vectorizer_object"
            - "model_object"
            - "label_encoder"
        prediction_column_name (str, optional): 
            The name for the new column containing predictions. Defaults to "predictions".
        return_confidence (bool, optional):
            If True, adds a 'confidence' column showing the model's probability score 
            for the winning class. Defaults to True.

    Returns:
        pl.DataFrame: 
            The original DataFrame (converted to Polars) with the new prediction column(s) added.
    """

    # Convert pandas to Polars if needed
    if isinstance(new_data, pd.DataFrame):
        new_data = pl.from_pandas(new_data)
    elif not isinstance(new_data, pl.DataFrame):
        raise TypeError(f"Expected Polars or pandas DataFrame, got {type(new_data)}")

    # Drop nulls in the text column
    new_data = new_data.drop_nulls(subset=[text_column_name])
    texts = new_data[text_column_name].to_list()

    vectorizer = trained_results["vectorizer_object"]
    best_model = trained_results["model_object"]
    label_encoder = trained_results["label_encoder"]
    
    # Generate features safely
    if hasattr(vectorizer, 'transform'):
        new_features = vectorizer.transform(texts)
    else:
        def text_to_vector(text):
            words = text.split()
            vectors = [vectorizer[word] for word in words if word in vectorizer]
            return np.mean(vectors, axis=0) if vectors else np.zeros(vectorizer.vector_size)
        new_features = np.array([text_to_vector(text) for text in texts])
    
    # Keras/Sparse Safety Net
    if type(best_model).__name__ == "KerasClassifier" and issparse(new_features):
        new_features = new_features.toarray()
    
    # Get numerical predictions
    numeric_predictions = best_model.predict(new_features)
    
    # Convert to original labels
    predictions = label_encoder.inverse_transform(numeric_predictions)
    
    # Initialize the list of columns to add
    columns_to_add = [pl.Series(prediction_column_name, predictions)]

    # --- NEW: Confidence Scores ---
    if return_confidence:
        # Check if the model supports probability estimation
        if hasattr(best_model, "predict_proba"):
            # Get the probability matrix for all classes
            all_probs = best_model.predict_proba(new_features)
            
            # Extract the maximum probability (the winning class) for each row
            confidence_scores = np.max(all_probs, axis=1)
            
            # Append the new column to our list
            columns_to_add.append(pl.Series("confidence", confidence_scores))
        else:
            warnings.warn(
                f"\n[WARNING] The model '{type(best_model).__name__}' does not support probability scoring. "
                f"The confidence column will not be added."
            )

    # Add all new columns to the DataFrame simultaneously
    return new_data.with_columns(columns_to_add)