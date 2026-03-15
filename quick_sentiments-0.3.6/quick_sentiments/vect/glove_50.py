import numpy as np
from gensim.models import Word2Vec
import gensim.downloader as api

_loaded_word2vec_model_instance = None

def sentence_vector(words, mod):
    word_vectors = [mod[word] for word in words if word in mod]
    if len(word_vectors) == 0:
        return np.zeros(mod.vector_size)
    return np.mean(word_vectors, axis=0)

def vectorize_train(texts):
    """
    Accepts:
        texts: list of strings (full corpus)
    
    Returns:
        numpy array of shape (n_samples, embedding_dim)
    """

    global _loaded_word2vec_model_instance

    if _loaded_word2vec_model_instance is None:
        print("Loading pre-trained glove-twitter-50 model (this may take a few minutes)...")
        _loaded_word2vec_model_instance = api.load('glove-twitter-50')
        print("Word2Vec model loaded.")
    else:
        print("Using already loaded Word2Vec model.")

    tokenized = [sentence.split() for sentence in texts]

    X_features = np.array([
        sentence_vector(tokens, _loaded_word2vec_model_instance)
        for tokens in tokenized
    ])

    # For Word2Vec, the loaded model instance itself serves as the "fitted vectorizer object"
    # because it contains the vocabulary and embeddings needed to transform new data consistently.
    return X_features, _loaded_word2vec_model_instance, None # since we are using array, order is necessary, thus we need to return the model instance as well

def vectorize_test(texts, loaded_model, norm=None):
    print("Transforming test data using loaded Word2Vec model...")
    tokenized = [sentence.split() for sentence in texts]
    X_features = np.array([sentence_vector(tokens, loaded_model) for tokens in tokenized])
    return X_features