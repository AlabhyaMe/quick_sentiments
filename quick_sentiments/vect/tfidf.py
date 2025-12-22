# Vect/tfidf_vectorizer.py

from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize_train(texts: list[str]):
    """
    Generates TF-IDF features for the entire dataset.

    Args:
        texts (list[str]): List of all preprocessed documents.

    Returns:
        np.ndarray: TF-IDF feature matrix (dense).
    """
    print("   - Generating TF-IDF features...")

    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(texts)  # returns sparse matrix

    
    # The 'vectorizer' object (CountVectorizer instance) is returned here
    # because it is 'fitted' on the training data and contains the learned
    # vocabulary. This fitted object is essential to consistently transform
    # new, unseen data into the same feature space.
    
    return X_tfidf, vectorizer, None

def vectorize_test(texts, fitted_vectorizer,norm=None):
    """
    Transform test data using fitted Bag-of-Words vectorizer
    and generate normalized TF-IDF features.
    """
    print("   - Transforming test data using fitted Bag-of-Words vectorizer...")
    X_tfidf = fitted_vectorizer.transform(texts)  # SAME TF-IDF space
    return X_tfidf
