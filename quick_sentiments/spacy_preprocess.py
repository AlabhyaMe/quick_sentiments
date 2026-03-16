import polars as pl
import pandas as pd
import spacy
import subprocess
import sys
from typing import Union

# --- 1. GLOBAL SETUP (Prevents Crashes) ---

try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def pre_process_spacy(
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

    df = df.with_columns(expr.alias(new_column_name))

# --- SLOW PYTHON STEPS (Conditional, optimized with spaCy) ---
    if remove_stop_words or lemmatize:
        # 1. Extract the column to a standard Python list
        texts = df[new_column_name].to_list()
        processed_texts = []
        
   
        # 2. Use nlp.pipe() for bulk processing (much faster than row-by-row)
        for doc in nlp.pipe(texts, batch_size=1000):
            tokens = []
            for token in doc:
                # Check for stop words
                if remove_stop_words and token.is_stop:
                    continue
                
                # Lemmatize or keep text
                if lemmatize:
                    tokens.append(token.lemma_)
                else:
                    tokens.append(token.text)
                    
            # Rejoin tokens and add to our new list
            processed_texts.append(" ".join(tokens))
            
        # 3. Overwrite the Polars column with our newly processed list
        df = df.with_columns(pl.Series(new_column_name, processed_texts))

    return df