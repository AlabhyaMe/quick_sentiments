# quick_sentiments/__init__.py
from .pipeline import run_pipeline       # Expose pipeline function
from .predict import make_predictions # Expose prediction function
from .preprocess import pre_process_nltk  # Expose NLTK preprocessing function
from .spacy_preprocess import pre_process_spacy # Expose spaCy preprocessing function
from .evaluate_performance import evaluate_performance # Expose evaluation function
__all__ = [
    'run_pipeline', 
    'make_predictions',
    'pre_process_nltk',
    'pre_process_spacy',
    'evaluate_performance'
]

