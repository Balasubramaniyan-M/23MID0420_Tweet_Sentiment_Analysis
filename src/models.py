"""Classical ML model pipelines for tweet sentiment classification.

All models use sklearn Pipelines with TF-IDF vectorization to prevent
data leakage. TF-IDF is ALWAYS fitted only on training data.

Models:
- Logistic Regression (TF-IDF → LR)
- LinearSVC (TF-IDF → SVC)
- MultinomialNB (TF-IDF → NB)
"""

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, make_scorer, precision_score, recall_score,
)

from src.utils import SEED, N_FOLDS, SENTIMENT_LABELS, print_section
from src.preprocessing import TweetPreprocessor


def create_tfidf_lr_pipeline(
    strategy: str = "minimal",
    max_features: int = 50000,
    ngram_range: Tuple[int, int] = (1, 2),
    C: float = 1.0,
    max_iter: int = 1000,
) -> Pipeline:
    """Create TF-IDF → Logistic Regression pipeline.
    
    Parameters
    ----------
    strategy : str
        Preprocessing strategy ('minimal' or 'aggressive').
    max_features : int
        Maximum vocabulary size for TF-IDF.
    ngram_range : tuple
        N-gram range for TF-IDF.
    C : float
        Regularization strength for Logistic Regression.
    max_iter : int
        Maximum iterations for convergence.
    
    Returns
    -------
    Pipeline
        Complete preprocessing + vectorization + classification pipeline.
    """
    return Pipeline([
        ("preprocessor", TweetPreprocessor(strategy=strategy)),
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            min_df=2,
            strip_accents="unicode",
        )),
        ("clf", LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=SEED,
            solver="lbfgs",
            class_weight="balanced",
            n_jobs=-1,
        )),
    ])


def create_tfidf_svc_pipeline(
    strategy: str = "minimal",
    max_features: int = 50000,
    ngram_range: Tuple[int, int] = (1, 2),
    C: float = 1.0,
) -> Pipeline:
    """Create TF-IDF → LinearSVC pipeline.
    
    Uses CalibratedClassifierCV to enable probability estimates.
    
    Parameters
    ----------
    strategy : str
        Preprocessing strategy.
    max_features : int
        Maximum vocabulary size.
    ngram_range : tuple
        N-gram range.
    C : float
        Regularization parameter.
    
    Returns
    -------
    Pipeline
        Complete pipeline with calibrated SVC.
    """
    return Pipeline([
        ("preprocessor", TweetPreprocessor(strategy=strategy)),
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            min_df=2,
            strip_accents="unicode",
        )),
        ("clf", CalibratedClassifierCV(
            estimator=LinearSVC(
                C=C,
                random_state=SEED,
                max_iter=2000,
                class_weight="balanced",
            ),
            cv=3,
        )),
    ])


def create_tfidf_nb_pipeline(
    strategy: str = "minimal",
    max_features: int = 50000,
    ngram_range: Tuple[int, int] = (1, 2),
    alpha: float = 1.0,
) -> Pipeline:
    """Create TF-IDF → MultinomialNB pipeline.
    
    Parameters
    ----------
    strategy : str
        Preprocessing strategy.
    max_features : int
        Maximum vocabulary size.
    ngram_range : tuple
        N-gram range.
    alpha : float
        Additive smoothing parameter.
    
    Returns
    -------
    Pipeline
        Complete pipeline.
    """
    return Pipeline([
        ("preprocessor", TweetPreprocessor(strategy=strategy)),
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            min_df=2,
            strip_accents="unicode",
        )),
        ("clf", MultinomialNB(alpha=alpha)),
    ])


def get_all_pipelines(strategy: str = "minimal") -> Dict[str, Pipeline]:
    """Return a dictionary of all model pipelines."""
    return {
        "LogisticRegression": create_tfidf_lr_pipeline(strategy=strategy),
        "LinearSVC": create_tfidf_svc_pipeline(strategy=strategy),
        "MultinomialNB": create_tfidf_nb_pipeline(strategy=strategy),
    }


def run_cross_validation(
    pipelines: Dict[str, Pipeline],
    X_train: pd.Series,
    y_train: pd.Series,
    n_folds: int = N_FOLDS,
    seed: int = SEED,
) -> pd.DataFrame:
    """Run stratified k-fold cross-validation on all pipelines.
    
    Parameters
    ----------
    pipelines : dict
        {name: pipeline} dictionary.
    X_train : pd.Series
        Training text data.
    y_train : pd.Series
        Training labels.
    n_folds : int
        Number of CV folds.
    seed : int
        Random state.
    
    Returns
    -------
    pd.DataFrame
        CV results with mean and std for each metric.
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    scoring = {
        "accuracy": "accuracy",
        "macro_f1": "f1_macro",
        "weighted_f1": "f1_weighted",
        "macro_precision": "precision_macro",
        "macro_recall": "recall_macro",
    }
    
    results = []
    for name, pipeline in pipelines.items():
        print(f"\n  Cross-validating: {name}")
        start_time = time.time()
        
        cv_results = cross_validate(
            pipeline, X_train, y_train,
            cv=cv, scoring=scoring,
            return_train_score=False,
            n_jobs=-1,
        )
        
        elapsed = time.time() - start_time
        
        result = {
            "model": name,
            "cv_folds": n_folds,
            "mean_accuracy": np.mean(cv_results["test_accuracy"]),
            "std_accuracy": np.std(cv_results["test_accuracy"]),
            "mean_macro_f1": np.mean(cv_results["test_macro_f1"]),
            "std_macro_f1": np.std(cv_results["test_macro_f1"]),
            "mean_weighted_f1": np.mean(cv_results["test_weighted_f1"]),
            "std_weighted_f1": np.std(cv_results["test_weighted_f1"]),
            "mean_macro_precision": np.mean(cv_results["test_macro_precision"]),
            "std_macro_precision": np.std(cv_results["test_macro_precision"]),
            "mean_macro_recall": np.mean(cv_results["test_macro_recall"]),
            "std_macro_recall": np.std(cv_results["test_macro_recall"]),
            "total_cv_time_seconds": round(elapsed, 2),
        }
        results.append(result)
        
        print(f"    Mean Macro F1: {result['mean_macro_f1']:.4f} ± {result['std_macro_f1']:.4f}")
        print(f"    Mean Accuracy: {result['mean_accuracy']:.4f} ± {result['std_accuracy']:.4f}")
        print(f"    Time: {elapsed:.2f}s")
    
    return pd.DataFrame(results)


def train_final_model(
    pipeline: Pipeline,
    X_train: pd.Series,
    y_train: pd.Series,
) -> Pipeline:
    """Train the selected model on the full training set.
    
    Parameters
    ----------
    pipeline : Pipeline
        The selected model pipeline.
    X_train : pd.Series
        Full training text data (train + val).
    y_train : pd.Series
        Full training labels.
    
    Returns
    -------
    Pipeline
        The fitted pipeline.
    """
    print("  Training final model on full training data...")
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    elapsed = time.time() - start_time
    print(f"  ⏱ Training completed in {elapsed:.2f}s")
    return pipeline


def evaluate_on_test(
    pipeline: Pipeline,
    X_test: pd.Series,
    y_test: pd.Series,
    model_name: str = "SelectedModel",
) -> Dict:
    """Evaluate a fitted pipeline on the locked test set.
    
    Parameters
    ----------
    pipeline : Pipeline
        Fitted model pipeline.
    X_test : pd.Series
        Test text data.
    y_test : pd.Series
        Test labels.
    model_name : str
        Model identifier for reporting.
    
    Returns
    -------
    Dict
        Comprehensive test metrics.
    """
    labels = SENTIMENT_LABELS
    
    # Predictions
    start_time = time.time()
    y_pred = pipeline.predict(X_test)
    inference_time = time.time() - start_time
    
    # Probabilities (if available)
    y_proba = None
    if hasattr(pipeline, "predict_proba"):
        try:
            y_proba = pipeline.predict_proba(X_test)
        except Exception:
            pass
    
    # Metrics
    metrics = {
        "model_name": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_precision": precision_score(y_test, y_pred, average="macro", labels=labels, zero_division=0),
        "macro_recall": recall_score(y_test, y_pred, average="macro", labels=labels, zero_division=0),
        "macro_f1": f1_score(y_test, y_pred, average="macro", labels=labels, zero_division=0),
        "weighted_f1": f1_score(y_test, y_pred, average="weighted", labels=labels, zero_division=0),
        "classification_report": classification_report(
            y_test, y_pred, labels=labels, output_dict=True, zero_division=0
        ),
        "classification_report_str": classification_report(
            y_test, y_pred, labels=labels, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels),
        "predictions": y_pred,
        "probabilities": y_proba,
        "inference_time_seconds": round(inference_time, 4),
        "throughput_tweets_per_second": round(len(X_test) / inference_time, 1),
    }
    
    print(f"\n{'='*50}")
    print(f"  LOCKED TEST SET RESULTS: {model_name}")
    print(f"{'='*50}")
    print(f"  Accuracy:        {metrics['accuracy']:.4f}")
    print(f"  Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall:    {metrics['macro_recall']:.4f}")
    print(f"  Macro F1:        {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1:     {metrics['weighted_f1']:.4f}")
    print(f"  Inference:       {metrics['inference_time_seconds']:.4f}s")
    print(f"  Throughput:      {metrics['throughput_tweets_per_second']:.1f} tweets/s")
    print(f"\n{metrics['classification_report_str']}")
    
    return metrics


def predict_sentiment(
    pipeline: Pipeline,
    text: str,
    preprocess: bool = False,
) -> Dict:
    """Predict sentiment for a single tweet.
    
    Parameters
    ----------
    pipeline : Pipeline
        Fitted pipeline (includes preprocessing).
    text : str
        Input tweet text.
    preprocess : bool
        Whether pipeline already includes preprocessing step.
    
    Returns
    -------
    Dict
        Prediction result with label and confidence.
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return {
            "input": text,
            "predicted_sentiment": None,
            "confidence": None,
            "note": "Invalid or empty input.",
        }
    
    prediction = pipeline.predict([text])[0]
    
    result = {
        "input": text,
        "predicted_sentiment": prediction,
        "confidence": None,
        "class_probabilities": None,
        "note": (
            "This is an analytical signal, not objective truth. "
            "Model probabilities may not be well-calibrated. "
            "Human oversight is recommended before consequential decisions."
        ),
    }
    
    if hasattr(pipeline, "predict_proba"):
        try:
            proba = pipeline.predict_proba([text])[0]
            result["confidence"] = round(float(max(proba)), 4)
            result["class_probabilities"] = {
                label: round(float(p), 4)
                for label, p in zip(pipeline.classes_, proba)
            }
        except Exception:
            pass
    
    return result
