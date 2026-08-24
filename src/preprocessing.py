"""Tweet-specific text preprocessing.

Implements minimal, justified normalization that preserves
sentiment-bearing features (negation, emojis, punctuation)
while standardizing noise (URLs, mentions, whitespace).

Two preprocessing strategies are provided for ablation:
- minimal_preprocess: preserves hashtags, emojis, punctuation
- aggressive_preprocess: removes hashtags, emojis, extra punctuation
"""

import re
from typing import Callable, List, Optional

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def replace_urls(text: str) -> str:
    """Replace URLs with <URL> token."""
    return re.sub(r'https?://\S+|www\.\S+', '<URL>', text)


def replace_mentions(text: str) -> str:
    """Replace @username mentions with <USER> token."""
    return re.sub(r'@\w+', '<USER>', text)


def normalize_whitespace(text: str) -> str:
    """Collapse multiple whitespace into single spaces and strip."""
    return re.sub(r'\s+', ' ', text).strip()


def normalize_repeated_chars(text: str) -> str:
    """Reduce character repetitions to max 3 (e.g., 'sooooo' → 'sooo')."""
    return re.sub(r'(.)\1{3,}', r'\1\1\1', text)


def remove_hashtag_symbol(text: str) -> str:
    """Remove # symbol but keep the hashtag word."""
    return re.sub(r'#(\w+)', r'\1', text)


def remove_emojis(text: str) -> str:
    """Remove emoji characters."""
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub('', text)


def remove_extra_punctuation(text: str) -> str:
    """Remove repeated punctuation (e.g., '!!!' → '!')."""
    text = re.sub(r'([!?.]){2,}', r'\1', text)
    return text


def lowercase(text: str) -> str:
    """Convert text to lowercase."""
    return text.lower()


def minimal_preprocess(text: str) -> str:
    """Apply minimal tweet preprocessing.
    
    Preserves sentiment-bearing features:
    - Emojis (sentiment signals)
    - Hashtags (topic and sentiment)
    - Exclamation/question marks (intensity)
    - Negation words
    - Case variation (some signal)
    
    Only normalizes noise:
    - URLs → <URL>
    - Mentions → <USER>
    - Excessive character repetition
    - Whitespace normalization
    - Lowercase (for TF-IDF consistency)
    """
    if not isinstance(text, str):
        return ""
    text = replace_urls(text)
    text = replace_mentions(text)
    text = normalize_repeated_chars(text)
    text = lowercase(text)
    text = normalize_whitespace(text)
    return text


def aggressive_preprocess(text: str) -> str:
    """Apply aggressive tweet preprocessing (for ablation study).
    
    Removes features that may carry sentiment information:
    - Emojis removed
    - Hashtag symbols removed
    - Repeated punctuation collapsed
    - Same URL/mention/whitespace normalization as minimal
    """
    if not isinstance(text, str):
        return ""
    text = replace_urls(text)
    text = replace_mentions(text)
    text = remove_emojis(text)
    text = remove_hashtag_symbol(text)
    text = remove_extra_punctuation(text)
    text = normalize_repeated_chars(text)
    text = lowercase(text)
    text = normalize_whitespace(text)
    return text


class TweetPreprocessor(BaseEstimator, TransformerMixin):
    """Sklearn-compatible tweet preprocessor for pipeline integration.
    
    Parameters
    ----------
    strategy : str
        Preprocessing strategy: 'minimal' or 'aggressive'.
    """
    
    def __init__(self, strategy: str = "minimal"):
        self.strategy = strategy
        self._preprocessor: Callable = (
            minimal_preprocess if strategy == "minimal"
            else aggressive_preprocess
        )
    
    def fit(self, X, y=None):
        """No fitting required for rule-based preprocessing."""
        return self
    
    def transform(self, X) -> List[str]:
        """Apply preprocessing to text data.
        
        Parameters
        ----------
        X : array-like of str
            Raw tweet texts.
        
        Returns
        -------
        List[str]
            Preprocessed tweet texts.
        """
        if isinstance(X, pd.Series):
            return X.apply(self._preprocessor).tolist()
        return [self._preprocessor(text) for text in X]
    
    def get_params(self, deep=True):
        return {"strategy": self.strategy}
    
    def set_params(self, **params):
        if "strategy" in params:
            self.strategy = params["strategy"]
            self._preprocessor = (
                minimal_preprocess if self.strategy == "minimal"
                else aggressive_preprocess
            )
        return self


def preprocess_dataframe(
    df: pd.DataFrame,
    text_column: str = "text",
    strategy: str = "minimal",
    output_column: str = "text_clean",
) -> pd.DataFrame:
    """Apply preprocessing to a dataframe's text column.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    text_column : str
        Name of the text column.
    strategy : str
        Preprocessing strategy: 'minimal' or 'aggressive'.
    output_column : str
        Name for the preprocessed text column.
    
    Returns
    -------
    pd.DataFrame
        Dataframe with added preprocessed column.
    """
    preprocess_fn = minimal_preprocess if strategy == "minimal" else aggressive_preprocess
    df = df.copy()
    df[output_column] = df[text_column].apply(preprocess_fn)
    return df


def get_preprocessing_summary(original: pd.Series, processed: pd.Series) -> dict:
    """Compare original vs preprocessed text statistics."""
    return {
        "original_avg_length": round(original.str.len().mean(), 1),
        "processed_avg_length": round(processed.str.len().mean(), 1),
        "original_avg_words": round(original.str.split().str.len().mean(), 1),
        "processed_avg_words": round(processed.str.split().str.len().mean(), 1),
        "urls_found": int(original.str.contains(r'https?://\S+', regex=True).sum()),
        "mentions_found": int(original.str.contains(r'@\w+', regex=True).sum()),
    }
