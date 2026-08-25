"""Baseline classifiers: DummyClassifier and VADER.

Provides the performance floor (Dummy) and unsupervised
social-media sentiment baseline (VADER) for comparison
against trained classical models.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.utils import SEED, SENTIMENT_LABELS, print_section


def train_dummy_baseline(
    y_train: pd.Series,
    y_test: pd.Series,
    strategy: str = "most_frequent",
) -> Dict:
    """Train and evaluate a DummyClassifier baseline.
    
    Parameters
    ----------
    y_train : pd.Series
        Training target labels.
    y_test : pd.Series
        Test target labels.
    strategy : str
        DummyClassifier strategy (default: 'most_frequent').
    
    Returns
    -------
    Dict
        Dictionary with model, predictions, and all metrics.
    """
    dummy = DummyClassifier(strategy=strategy, random_state=SEED)
    dummy.fit(y_train.values.reshape(-1, 1), y_train)  # X is irrelevant for dummy
    y_pred = dummy.predict(y_test.values.reshape(-1, 1))
    
    metrics = compute_baseline_metrics(y_test, y_pred, "DummyClassifier")
    metrics["model"] = dummy
    metrics["predictions"] = y_pred
    metrics["strategy"] = strategy
    return metrics


def vader_predict(texts: pd.Series) -> List[str]:
    """Predict sentiment using VADER.
    
    Maps VADER compound scores to 3-class sentiment:
    - compound >= 0.05  → positive
    - compound <= -0.05 → negative  
    - else              → neutral
    
    Parameters
    ----------
    texts : pd.Series
        Raw tweet texts.
    
    Returns
    -------
    List[str]
        Predicted sentiment labels.
    """
    analyzer = SentimentIntensityAnalyzer()
    predictions = []
    scores = []
    
    for text in texts:
        if not isinstance(text, str):
            text = ""
        vs = analyzer.polarity_scores(text)
        scores.append(vs)
        compound = vs["compound"]
        if compound >= 0.05:
            predictions.append("positive")
        elif compound <= -0.05:
            predictions.append("negative")
        else:
            predictions.append("neutral")
    
    return predictions, scores


def evaluate_vader_baseline(
    texts: pd.Series,
    y_true: pd.Series,
) -> Dict:
    """Evaluate VADER on the given text and true labels.
    
    Parameters
    ----------
    texts : pd.Series
        Raw tweet texts (unprocessed, as VADER has its own lexicon).
    y_true : pd.Series
        True sentiment labels.
    
    Returns
    -------
    Dict
        Dictionary with predictions, compound scores, and all metrics.
    """
    predictions, scores = vader_predict(texts)
    y_pred = np.array(predictions)
    compound_scores = [s["compound"] for s in scores]
    
    metrics = compute_baseline_metrics(y_true, y_pred, "VADER")
    metrics["predictions"] = y_pred
    metrics["compound_scores"] = compound_scores
    metrics["vader_scores"] = scores
    return metrics


def compute_baseline_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
) -> Dict:
    """Compute comprehensive classification metrics.
    
    Parameters
    ----------
    y_true : pd.Series
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    model_name : str
        Name for display.
    
    Returns
    -------
    Dict
        All computed metrics.
    """
    labels = SENTIMENT_LABELS
    
    metrics = {
        "model_name": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0),
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        ),
        "classification_report_str": classification_report(
            y_true, y_pred, labels=labels, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
    }
    
    print(f"\n{model_name} Results:")
    print(f"  Accuracy:        {metrics['accuracy']:.4f}")
    print(f"  Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall:    {metrics['macro_recall']:.4f}")
    print(f"  Macro F1:        {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1:     {metrics['weighted_f1']:.4f}")
    print(f"\n{metrics['classification_report_str']}")
    return metrics
