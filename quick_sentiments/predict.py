from typing import Union, Dict, Any
import polars as pl
import pandas as pd
import numpy as np

def make_predictions(
        new_data: Union[pl.DataFrame, pd.DataFrame],
        text_column_name: str,
        trained_results: Dict[str, Any],
        prediction_column_name: str = "predictions") -> pl.DataFrame:
    """
    Generates predictions for new data using the results dictionary from run_pipeline.

    This function automatically extracts the vectorizer, trained model, and label 
    encoder from the `trained_results` dictionary. It handles missing data, 
    text vectorization, and decodes numerical predictions back into human-readable 
    labels (e.g., 'positive', 'negative').

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

    Returns:
        pl.DataFrame: 
            The original DataFrame (converted to Polars) with the new prediction column added.
            Rows with missing text (None/NaN) in the target column will be dropped.

    Example:
        >>> # Get results from the pipeline
        >>> results = run_pipeline(df_train, "text", "label")
        >>> 
        >>> # Predict on new data using the whole results object
        >>> preds = make_predictions(
        ...     new_data=df_new,
        ...     text_column_name="clean_text",
        ...     trained_results=results
        ... )
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
    
    # Generate features
    if hasattr(vectorizer, 'transform'):
        new_features = vectorizer.transform(texts)
    else:
        def text_to_vector(text):
            words = text.split()
            vectors = [vectorizer[word] for word in words if word in vectorizer]
            return np.mean(vectors, axis=0) if vectors else np.zeros(vectorizer.vector_size)
        new_features = np.array([text_to_vector(text) for text in texts])
    
    # Get numerical predictions
    numeric_predictions = best_model.predict(new_features)
    
    # Convert to original labels
    predictions = label_encoder.inverse_transform(numeric_predictions)
    
    # Add predictions as new column
    return new_data.with_columns(
        pl.Series(prediction_column_name, predictions)
    )