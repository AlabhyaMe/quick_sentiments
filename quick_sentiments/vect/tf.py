from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Normalizer
import numpy as np

def vectorize_train(texts):
    """
    Generates pure Term Frequency (TF) features for the entire dataset.

    Args:
        texts (list[str]): List of all preprocessed documents.

    Returns:
        np.ndarray: TF feature matrix (dense).
    """
    print("   - Generating TF features...")

    # Step 1 - Fit BoW model (Returns a sparse CSR matrix)
    vectorizer = CountVectorizer()
    X_counts = vectorizer.fit_transform(texts)

    # Step 2 - Normalize by row sum to get TF
    # Normalizer with norm='l1' is mathematically identical to 
    # (row / row_sum), but it keeps the data in sparse format.
    normalizer = Normalizer(norm='l1')
    X_tf = normalizer.fit_transform(X_counts)
        
    # The 'vectorizer' object (CountVectorizer instance) is returned here
    # because it is 'fitted' on the training data and contains the learned
    # vocabulary. This fitted object is essential to consistently transform
    # new, unseen data into the same feature space.
    return X_tf, vectorizer, normalizer


def vectorize_test(texts, fitted_vectorizer,fitted_normalizer):
    """
    Transform test data using fitted Bag-of-Words vectorizer
    and generate normalized Term Frequency (TF) features.
    """
    print("   - Transforming test data using fitted Bag-of-Words vectorizer...")
    # Transform to counts (sparse)
    X_counts = fitted_vectorizer.transform(texts)
    # Normalize (sparse)
    X_tf = fitted_normalizer.transform(X_counts)
    
    return X_tf
