"""Evaluation utilities for model comparison and test assessment.

Provides functions for saving metrics, creating comparison tables,
and generating evaluation artifacts.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)

from src.utils import (
    RESULTS_DIR, ARTIFACTS_DIR, MODELS_DIR, SENTIMENT_LABELS,
    save_json, print_section,
)


def save_cv_results(cv_df: pd.DataFrame, path: Optional[Path] = None) -> Path:
    """Save cross-validation results to CSV."""
    if path is None:
        path = RESULTS_DIR / "cv_results.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    cv_df.to_csv(path, index=False)
    print(f"Saved CV results: {path}")
    return path


def save_test_metrics(metrics: Dict, path: Optional[Path] = None) -> Path:
    """Save final test metrics to CSV."""
    if path is None:
        path = RESULTS_DIR / "final_test_metrics.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Flatten metrics for CSV
    flat = {
        "model_name": metrics["model_name"],
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "inference_time_seconds": metrics.get("inference_time_seconds"),
        "throughput_tweets_per_second": metrics.get("throughput_tweets_per_second"),
    }
    pd.DataFrame([flat]).to_csv(path, index=False)
    print(f"Saved test metrics: {path}")
    return path


def save_classification_report(
    y_true: pd.Series,
    y_pred: np.ndarray,
    path: Optional[Path] = None,
) -> Path:
    """Save per-class classification report to CSV."""
    if path is None:
        path = RESULTS_DIR / "classification_report.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    
    report = classification_report(
        y_true, y_pred, labels=SENTIMENT_LABELS,
        output_dict=True, zero_division=0,
    )
    report_df = pd.DataFrame(report).T
    report_df.to_csv(path)
    print(f"Saved classification report: {path}")
    return path


def save_test_predictions(
    test_df: pd.DataFrame,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    path: Optional[Path] = None,
) -> Path:
    """Save test predictions with actual labels."""
    if path is None:
        path = RESULTS_DIR / "test_predictions.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    
    pred_df = pd.DataFrame({
        "index": test_df.index,
        "actual": test_df.iloc[:, 0] if len(test_df.columns) == 1 else test_df.values.ravel(),
        "predicted": y_pred,
        "correct": (test_df.values.ravel() == y_pred) if hasattr(test_df, 'values') else None,
    })
    
    if y_proba is not None:
        for i, label in enumerate(SENTIMENT_LABELS):
            pred_df[f"prob_{label}"] = y_proba[:, i]
        pred_df["max_confidence"] = y_proba.max(axis=1)
    
    pred_df.to_csv(path, index=False)
    print(f"Saved test predictions: {path}")
    return path


def save_predictions_simple(
    y_true: pd.Series,
    y_pred: np.ndarray,
    texts: pd.Series,
    y_proba: Optional[np.ndarray] = None,
    path: Optional[Path] = None,
) -> Path:
    """Save test predictions in a clean format with text and labels."""
    if path is None:
        path = RESULTS_DIR / "test_predictions.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    
    pred_df = pd.DataFrame({
        "text": texts.values,
        "actual": y_true.values,
        "predicted": y_pred,
        "correct": y_true.values == y_pred,
    })
    
    if y_proba is not None:
        for i, label in enumerate(SENTIMENT_LABELS):
            pred_df[f"prob_{label}"] = y_proba[:, i]
        pred_df["max_confidence"] = y_proba.max(axis=1)
    
    pred_df.to_csv(path, index=False)
    print(f"Saved test predictions: {path}")
    return path


def select_best_model(
    cv_df: pd.DataFrame,
    primary_metric: str = "mean_macro_f1",
) -> Dict:
    """Select the best model based on CV results.
    
    Selection criteria (in order):
    1. Highest mean macro F1
    2. Lowest std macro F1 (stability)
    3. Runtime consideration
    
    Parameters
    ----------
    cv_df : pd.DataFrame
        Cross-validation results dataframe.
    primary_metric : str
        Column name for the primary selection metric.
    
    Returns
    -------
    Dict
        Selection decision with justification.
    """
    # Sort by primary metric descending, then by stability (lower std)
    ranked = cv_df.sort_values(
        by=[primary_metric, "std_macro_f1"],
        ascending=[False, True],
    ).reset_index(drop=True)
    
    best = ranked.iloc[0]
    
    selection = {
        "selected_model": best["model"],
        "selection_metric": primary_metric,
        "selected_value": round(best[primary_metric], 4),
        "selected_std": round(best["std_macro_f1"], 4),
        "ranking": ranked[["model", primary_metric, "std_macro_f1"]].to_dict(orient="records"),
        "justification": (
            f"{best['model']} was selected because it achieved the highest "
            f"mean macro F1 ({best[primary_metric]:.4f} ± {best['std_macro_f1']:.4f}) "
            f"in {int(best['cv_folds'])}-fold stratified cross-validation on training data only. "
            f"The final test set was not used for model selection."
        ),
    }
    
    print(f"\n  Selected Model: {selection['selected_model']}")
    print(f"  Justification: {selection['justification']}")
    
    return selection


def create_comparison_table(
    baseline_results: Dict[str, Dict],
    cv_df: pd.DataFrame,
    test_metrics: Optional[Dict] = None,
) -> pd.DataFrame:
    """Create a comprehensive model comparison table."""
    rows = []
    
    # Baselines
    for name, result in baseline_results.items():
        rows.append({
            "Model": name,
            "Type": "Baseline",
            "Accuracy": round(result["accuracy"], 4),
            "Macro F1": round(result["macro_f1"], 4),
            "Weighted F1": round(result["weighted_f1"], 4),
            "Macro Precision": round(result["macro_precision"], 4),
            "Macro Recall": round(result["macro_recall"], 4),
        })
    
    # CV models
    for _, row in cv_df.iterrows():
        rows.append({
            "Model": row["model"],
            "Type": "Trained (CV)",
            "Accuracy": round(row["mean_accuracy"], 4),
            "Macro F1": round(row["mean_macro_f1"], 4),
            "Weighted F1": round(row["mean_weighted_f1"], 4),
            "Macro Precision": round(row["mean_macro_precision"], 4),
            "Macro Recall": round(row["mean_macro_recall"], 4),
        })
    
    return pd.DataFrame(rows)


def save_experiment_manifest(
    env_info: Dict,
    dataset_info: Dict,
    split_info: Dict,
    cv_results: pd.DataFrame,
    selection: Dict,
    test_metrics: Dict,
) -> Path:
    """Save comprehensive experiment manifest for reproducibility."""
    manifest = {
        "environment": env_info,
        "dataset": {
            "name": dataset_info.get("dataset_name"),
            "source": dataset_info.get("url"),
            "n_rows": dataset_info.get("n_rows"),
            "n_columns": dataset_info.get("n_columns"),
        },
        "split": split_info,
        "cv_summary": cv_results.to_dict(orient="records"),
        "model_selection": selection,
        "test_results": {
            "model": test_metrics["model_name"],
            "accuracy": test_metrics["accuracy"],
            "macro_f1": test_metrics["macro_f1"],
            "weighted_f1": test_metrics["weighted_f1"],
        },
    }
    path = ARTIFACTS_DIR / "experiment_manifest.json"
    save_json(manifest, path)
    return path
