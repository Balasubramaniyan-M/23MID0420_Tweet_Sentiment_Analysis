"""Error analysis for tweet sentiment classification.

Inspects misclassified tweets to identify linguistic patterns,
business implications, and model limitations.
"""

import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils import RESULTS_DIR, SENTIMENT_LABELS, print_section


def extract_errors(
    texts: pd.Series,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    entities: Optional[pd.Series] = None,
    n_errors: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Extract misclassified examples for analysis.
    
    Parameters
    ----------
    texts : pd.Series
        Original tweet texts.
    y_true : pd.Series
        True sentiment labels.
    y_pred : np.ndarray
        Predicted sentiment labels.
    y_proba : np.ndarray, optional
        Prediction probabilities.
    entities : pd.Series, optional
        Entity/airline column.
    n_errors : int
        Number of errors to extract.
    seed : int
        Random seed for sampling.
    
    Returns
    -------
    pd.DataFrame
        DataFrame of misclassified examples with analysis metadata.
    """
    # Identify errors
    error_mask = y_true.values != y_pred
    error_indices = np.where(error_mask)[0]
    
    if len(error_indices) == 0:
        print("No errors found!")
        return pd.DataFrame()
    
    # Sample errors (stratified across error types if possible)
    rng = np.random.RandomState(seed)
    n_sample = min(n_errors, len(error_indices))
    sampled_indices = rng.choice(error_indices, size=n_sample, replace=False)
    sampled_indices = sorted(sampled_indices)
    
    # Build error dataframe
    records = []
    for idx in sampled_indices:
        record = {
            "index": int(idx),
            "text_anonymized": anonymize_tweet(texts.iloc[idx]),
            "actual_sentiment": y_true.iloc[idx],
            "predicted_sentiment": y_pred[idx],
            "error_type": f"{y_true.iloc[idx]} → {y_pred[idx]}",
        }
        
        if y_proba is not None:
            record["max_confidence"] = round(float(y_proba[idx].max()), 4)
            for i, label in enumerate(SENTIMENT_LABELS):
                record[f"prob_{label}"] = round(float(y_proba[idx][i]), 4)
        
        if entities is not None:
            record["entity"] = entities.iloc[idx]
        
        # Categorize error
        record["error_category"] = categorize_error(texts.iloc[idx])
        record["linguistic_notes"] = analyze_linguistic_features(texts.iloc[idx])
        
        records.append(record)
    
    error_df = pd.DataFrame(records)
    print(f"\nExtracted {len(error_df)} error examples for analysis")
    return error_df


def anonymize_tweet(text: str) -> str:
    """Remove potentially identifying information from a tweet.
    
    - Replace @mentions with <USER>
    - Remove URLs
    - Keep content for analysis
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r'@\w+', '<USER>', text)
    text = re.sub(r'https?://\S+', '<URL>', text)
    return text


def categorize_error(text: str) -> str:
    """Categorize the likely reason for misclassification."""
    if not isinstance(text, str):
        return "empty_text"
    
    text_lower = text.lower()
    
    # Check for various linguistic phenomena
    if any(word in text_lower for word in ["not", "n't", "never", "no ", "neither", "nor"]):
        return "negation"
    if any(marker in text_lower for word in ["but", "however", "although", "though"] for marker in [word]):
        if len(text) > 80:
            return "mixed_sentiment"
    if re.search(r'[😀-🙏🌀-🗿🚀-🤀-🧿]+', text):
        return "emoji_heavy"
    if any(word in text_lower for word in ["lol", "lmao", "smh", "fml", "ikr"]):
        return "slang_abbreviation"
    if re.search(r'#\w+', text):
        if text.count('#') >= 2:
            return "hashtag_heavy"
    if '"' in text or "'" in text:
        if any(word in text_lower for word in ["said", "says", "told", "quoted"]):
            return "quoted_content"
    if any(word in text_lower for word in ["yeah right", "sure", "great job", "thanks a lot"]):
        return "potential_sarcasm"
    if len(text.split()) <= 5:
        return "very_short"
    if len(text.split()) >= 25:
        return "long_complex"
    
    return "general_ambiguity"


def analyze_linguistic_features(text: str) -> str:
    """Generate linguistic notes about a tweet for error interpretation."""
    if not isinstance(text, str):
        return "Empty text"
    
    features = []
    text_lower = text.lower()
    
    # Check features
    if re.search(r"n't|not |never |no ", text_lower):
        features.append("contains negation")
    if text.count('!') >= 2:
        features.append(f"{text.count('!')} exclamation marks")
    if text.count('?') >= 1:
        features.append("contains question")
    mention_matches = re.findall(r'@\w+', text)
    if mention_matches:
        n_mentions = len(mention_matches)
        features.append(f"{n_mentions} mentions")
    hashtag_matches = re.findall(r'#\w+', text)
    if hashtag_matches:
        n_hashtags = len(hashtag_matches)
        features.append(f"{n_hashtags} hashtags")
    if text != text_lower and text != text.upper():
        caps = sum(1 for c in text if c.isupper())
        if caps > len(text) * 0.3:
            features.append("heavy caps usage")
    if len(text.split()) <= 4:
        features.append("very short tweet")
    
    return "; ".join(features) if features else "no notable features"


def compute_error_distribution(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """Compute the distribution of error types."""
    error_mask = y_true.values != y_pred
    
    error_pairs = pd.DataFrame({
        "actual": y_true.values[error_mask],
        "predicted": y_pred[error_mask],
    })
    error_pairs["error_type"] = error_pairs["actual"] + " → " + error_pairs["predicted"]
    
    distribution = error_pairs["error_type"].value_counts().reset_index()
    distribution.columns = ["error_type", "count"]
    distribution["percentage"] = (distribution["count"] / distribution["count"].sum() * 100).round(2)
    
    return distribution


def save_error_analysis(
    error_df: pd.DataFrame,
    error_distribution: pd.DataFrame,
    path: Optional[pd.core.frame.DataFrame] = None,
) -> None:
    """Save error analysis artifacts."""
    # Save detailed errors
    error_path = RESULTS_DIR / "error_analysis.csv"
    error_df.to_csv(error_path, index=False)
    print(f"Saved error analysis: {error_path}")
    
    # Save error distribution
    dist_path = RESULTS_DIR / "error_distribution.csv"
    error_distribution.to_csv(dist_path, index=False)
    print(f"Saved error distribution: {dist_path}")
