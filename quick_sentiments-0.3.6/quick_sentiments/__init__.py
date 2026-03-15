# quick_sentiments/__init__.py
from .pipeline import run_pipeline       # Expose pipeline function
from .predict import make_predictions # Expose prediction function
from .preprocess import pre_process  # Expose preprocessing function
from .spacy_preprocess import pre_process_spacy

__all__ = ['run_pipeline', 'make_predictions']  # Controls what's available in *

