from typing import Union
import polars as pl
import pandas as pd
import numpy as np

def make_predictions(
        new_data: Union[pl.DataFrame, pd.DataFrame],
        text_column_name: str,
        vectorizer,
        best_model,
        label_encoder,
        prediction_column_name: str = "predictions") -> pl.DataFrame:
    """
Generates predictions for new data using a trained model and vectorizer.

    This function handles the complexities of different vectorizers (Scikit-Learn vs. Gensim)
    and data formats (Pandas vs. Polars) automatically. It filters out missing data, 
    vectorizes the text, predicts using the model, and decodes the labels back to 
    their original string format (e.g., 'positive', 'negative').

    Args:
        new_data (Union[pl.DataFrame, pd.DataFrame]): 
            The input DataFrame containing the text to predict on.
        text_column_name (str): 
            The name of the column in `new_data` that contains the *preprocessed* text.
        vectorizer (Any): 
            The fitted vectorizer object. 
            - For BOW/TF-IDF: A Scikit-Learn vectorizer (must have `.transform()`).
            - For Word2Vec/GloVe: A Gensim KeyedVectors object.
        best_model (Any): 
            The trained machine learning model (e.g., RandomForest, XGBoost).
        label_encoder (Any): 
            The fitted LabelEncoder used during training to decode predictions 
            back to strings.
        prediction_column_name (str, optional): 
            The name for the new column containing predictions. Defaults to "predictions".

    Returns:
        pl.DataFrame: 
            The original DataFrame (converted to Polars) with the new prediction column added.
            Rows with missing text (None/NaN) in the target column will be dropped.

    Example:
        >>> # Assuming you have trained_results from run_pipeline()
        >>> preds = make_predictions(
        ...     new_data=df_new,
        ...     text_column_name="clean_text",
        ...     vectorizer=trained_results["vectorizer_object"],
        ...     best_model=trained_results["model_object"],
        ...     label_encoder=trained_results["label_encoder"]
        ... )
        >>> print(preds.select(["clean_text", "predictions"]))
        
    """
    # Convert pandas to Polars if needed
    if isinstance(new_data, pd.DataFrame):
        new_data = pl.from_pandas(new_data)
    elif not isinstance(new_data, pl.DataFrame):
        raise TypeError(f"Expected Polars or pandas DataFrame, got {type(new_data)}")

    # Drop nulls in the text column
    new_data = new_data.drop_nulls(subset=[text_column_name])
    texts = new_data[text_column_name].to_list()
    
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