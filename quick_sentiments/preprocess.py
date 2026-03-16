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


def pre_process_nltk(
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
    Cleans and preprocesses text data using fast Polars expressions.

    This function handles both Polars and pandas DataFrames. It prioritizes 
    speed by using native Rust-based string operations for cleaning (regex, 
    lowercase) and only falls back to Python for complex linguistic tasks.

    Args:
        df (Union[pl.DataFrame, pd.DataFrame]): 
            The input DataFrame containing the text to clean.
        text_column (str): 
            The name of the column containing the raw text.
        new_column_name (str, optional): 
            The name for the output column. Defaults to "processed_text".
        to_lowercase (bool, optional): 
            Convert text to lowercase. Defaults to True.
        remove_brackets (bool, optional): 
            Remove content inside square brackets []. Defaults to True.
        remove_urls (bool, optional): 
            Remove URLs starting with http/www. Defaults to True.
        remove_html (bool, optional): 
            Remove HTML tags (<br>, <div>). Defaults to True.
        remove_digits (bool, optional): 
            Remove all numeric digits. Defaults to True.
        remove_punct (bool, optional): 
            Remove punctuation marks. Defaults to True.
        remove_stop_words (bool, optional): 
            Remove common English stop words (slower). Defaults to False.
        lemmatize (bool, optional): 
            Convert words to their root form (slower). Defaults to False.

    Returns:
        pl.DataFrame: A Polars DataFrame with the new cleaned column.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"text": ["I LOVE this!!! <br>", "Click http://site.com"]})
        >>> clean_df = pre_process(df, "text", remove_urls=True)
        >>> print(clean_df)
        shape: (2, 2)
        ┌───────────────────────┬────────────────┐
        │ text                  ┆ processed_text │
        │ ---                   ┆ ---            │
        │ str                   ┆ str            │
        ╞═══════════════════════╪════════════════╡
        │ I LOVE this!!! <br>   ┆ i love this    │
        │ Click http://site.com ┆ click          │
        └───────────────────────┴────────────────┘

    """
    
    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)
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