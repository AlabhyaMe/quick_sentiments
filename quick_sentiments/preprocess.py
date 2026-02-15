import polars as pl
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from typing import Union

# --- 1. GLOBAL SETUP (Prevents Crashes) ---
# Check and download NLTK resources immediately to avoid runtime errors
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

# Initialize these GLOBALLY so they are ready for Polars workers
STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

# Force "wake up" the lemmatizer to prevent LazyLoader errors
LEMMATIZER.lemmatize("warmup")
# ------------------------------------------

def _process_linguistic(text: str, remove_stop_words: bool, lemmatize: bool) -> str:
    """
    Helper function for slow linguistic tasks (Python loop).
    Only called if absolutely necessary.
    """
    if not text:
        return ""
    
    # Simple whitespace tokenization (fast enough for this stage)
    tokens = text.split()

    if remove_stop_words:
        tokens = [t for t in tokens if t not in STOP_WORDS]

    if lemmatize:
        tokens = [LEMMATIZER.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def pre_process(
    df: Union[pl.DataFrame, pd.DataFrame],
    text_column: str,
    new_column_name: str = "processed_text",
    # Fast Polars Operations
    to_lowercase: bool = True,
    remove_brackets: bool = True,
    remove_urls: bool = True,
    remove_html: bool = True,
    remove_digits: bool = True,
    remove_punct: bool = True,
    # Slow Python Operations
    remove_stop_words: bool = False,
    lemmatize: bool = False
) -> pl.DataFrame:
    """
    Hybrid Preprocessing: 
    - Uses fast Polars expressions for string cleaning.
    - Uses Python fallback only for linguistic tasks (Stopwords/Lemmatization).
    """
    
    # 1. Start with Polars Expression (Handle Nulls safely)
    expr = pl.col(text_column).cast(pl.Utf8).fill_null("")

    # --- FAST NATIVE POLARS STEPS ---
    
    if to_lowercase:
        expr = expr.str.to_lowercase()

    if remove_brackets:
        expr = expr.str.replace_all(r"\[.*?\]", " ")

    if remove_urls:
        expr = expr.str.replace_all(r"http\S+|www.\S+", " ")

    if remove_html:
        expr = expr.str.replace_all(r"<.*?>", " ")

    if remove_digits:
        expr = expr.str.replace_all(r"\d+", " ")

    if remove_punct:
        # Keep words and spaces, replace everything else
        expr = expr.str.replace_all(r"[^\w\s]", " ")

    # Cleanup extra whitespace (Native)
    expr = expr.str.replace_all(r"\s+", " ").str.strip_chars()

    # Apply the fast steps first
    df = df.with_columns(expr.alias(new_column_name))

    # --- SLOW PYTHON STEPS (Conditional) ---
    # Only run this expensive step if the user actually asked for it
    if remove_stop_words or lemmatize:
        # We map over the ALREADY cleaned column (much faster/safer)
        df = df.with_columns(
            pl.col(new_column_name).map_elements(
                lambda x: _process_linguistic(x, remove_stop_words, lemmatize),
                return_dtype=pl.Utf8
            )
        )

    return df