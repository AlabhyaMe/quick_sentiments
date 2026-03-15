# quick_sentiments/vect/huggingface.py

import numpy as np

class HuggingFaceVectorizer:
    """
    A wrapper class to make Hugging Face sentence-transformers compatible
    with scikit-learn style pipelines. It mimics the .transform() behavior.
    """
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        # We import inside the init so it only loads if the user actually requests it
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "To use Hugging Face models, you must install sentence-transformers. "
                "Run: pip install sentence-transformers" \
                "Note: This will also install PyTorch, which is required for the models to work."
            )
        
        print(f"   - Loading Hugging Face Transformer model: {model_name}...")
        self.model = SentenceTransformer(model_name)

    def transform(self, texts):
        """
        Converts a list of texts into dense transformer embeddings.
        """
        # SentenceTransformer automatically handles batching and tokenization!
        return self.model.encode(texts, show_progress_bar=False)


def vectorize_train(texts):
    """
    Generates Hugging Face Transformer features for the training dataset.

    Args:
        texts (list[str]): List of preprocessed documents.

    Returns:
        tuple: (X_features, vectorizer_object, norm)
    """
    print("   - Generating Hugging Face embeddings (this may take a moment)...")
    
    # Initialize our custom wrapper wrapper
    # 'all-MiniLM-L6-v2' is the gold-standard lightweight model for text embeddings
    vectorizer_train = HuggingFaceVectorizer('all-MiniLM-L6-v2')
    
    # Generate the embeddings
    X_features = vectorizer_train.transform(texts)

    # Return the exact format pipeline.py expects
    return X_features, vectorizer_train, None

def vectorize_test(texts, fitted_vectorizer, norm=None):
    """
    Transform test data using the loaded Hugging Face model.
    """
    print("   - Transforming test data using Hugging Face embeddings...")
    # Because of our wrapper class, we can just call .transform()
    X_features = fitted_vectorizer.transform(texts)
    
    return X_features