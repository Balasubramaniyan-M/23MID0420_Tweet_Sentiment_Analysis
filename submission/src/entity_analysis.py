"""Product/entity (airline) sentiment analysis.

Performs support-aware entity-level analysis of sentiment
distribution and classifier performance per airline.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
)

from src.utils import (
    ENTITY_COLUMN, TARGET_COLUMN, SENTIMENT_LABELS,
    MIN_ENTITY_SUPPORT, RESULTS_DIR, print_section,
)


def compute_entity_sentiment_distribution(
    df: pd.DataFrame,
    entity_col: str = ENTITY_COLUMN,
    target_col: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """Compute sentiment distribution per entity.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset with entity and sentiment columns.
    entity_col : str
        Entity column name.
    target_col : str
        Target sentiment column name.
    
    Returns
    -------
    pd.DataFrame
        Cross-tabulation of entity × sentiment with counts and percentages.
    """
    # Count matrix
    ct = pd.crosstab(
        df[entity_col], df[target_col],
        margins=True, margins_name="Total",
    )
    
    # Percentage matrix
    ct_pct = pd.crosstab(
        df[entity_col], df[target_col],
        normalize="index",
    ).round(4) * 100
    
    # Combine
    result = ct.copy()
    for col in SENTIMENT_LABELS:
        if col in ct_pct.columns:
            result[f"{col}_pct"] = None
            for idx in ct_pct.index:
                if idx in result.index:
                    result.loc[idx, f"{col}_pct"] = round(ct_pct.loc[idx, col], 2)
    
    return result


def compute_entity_performance(
    y_true: pd.Series,
    y_pred: np.ndarray,
    entities: pd.Series,
    min_support: int = MIN_ENTITY_SUPPORT,
) -> pd.DataFrame:
    """Compute per-entity classifier performance.
    
    Only reports entities with at least min_support test samples.
    
    Parameters
    ----------
    y_true : pd.Series
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    entities : pd.Series
        Entity labels per sample.
    min_support : int
        Minimum samples required per entity.
    
    Returns
    -------
    pd.DataFrame
        Per-entity performance metrics.
    """
    results = []
    
    for entity in sorted(entities.unique()):
        mask = entities == entity
        n = mask.sum()
        
        if n < min_support:
            continue
        
        y_t = y_true[mask]
        y_p = y_pred[mask]
        
        result = {
            "entity": entity,
            "n_samples": int(n),
            "accuracy": round(accuracy_score(y_t, y_p), 4),
            "error_rate": round(1 - accuracy_score(y_t, y_p), 4),
            "macro_f1": round(f1_score(y_t, y_p, average="macro", labels=SENTIMENT_LABELS, zero_division=0), 4),
            "weighted_f1": round(f1_score(y_t, y_p, average="weighted", labels=SENTIMENT_LABELS, zero_division=0), 4),
        }
        
        # Per-class recall
        for label in SENTIMENT_LABELS:
            label_mask = y_t == label
            if label_mask.sum() > 0:
                result[f"recall_{label}"] = round(
                    (y_p[label_mask] == label).mean(), 4
                )
            else:
                result[f"recall_{label}"] = None
        
        results.append(result)
    
    entity_df = pd.DataFrame(results)
    
    if len(entity_df) > 0:
        entity_df = entity_df.sort_values("macro_f1", ascending=False).reset_index(drop=True)
    
    return entity_df


def compute_entity_error_analysis(
    y_true: pd.Series,
    y_pred: np.ndarray,
    entities: pd.Series,
    min_support: int = MIN_ENTITY_SUPPORT,
) -> pd.DataFrame:
    """Analyze error patterns per entity."""
    results = []
    
    for entity in sorted(entities.unique()):
        mask = entities == entity
        n = mask.sum()
        
        if n < min_support:
            continue
        
        y_t = y_true[mask].values
        y_p = y_pred[mask]
        errors = y_t != y_p
        
        result = {
            "entity": entity,
            "n_total": int(n),
            "n_errors": int(errors.sum()),
            "error_rate": round(errors.mean(), 4),
        }
        
        # Common error types
        if errors.sum() > 0:
            error_pairs = pd.Series(
                [f"{a}→{p}" for a, p in zip(y_t[errors], y_p[errors])]
            )
            most_common = error_pairs.value_counts().head(3)
            result["top_error_types"] = most_common.to_dict()
        
        results.append(result)
    
    return pd.DataFrame(results)


def generate_entity_report(
    entity_sentiment: pd.DataFrame,
    entity_performance: pd.DataFrame,
    min_support: int = MIN_ENTITY_SUPPORT,
) -> str:
    """Generate a text summary of entity analysis."""
    lines = [
        f"Entity Analysis Summary (min. support: {min_support} test samples)",
        "=" * 60,
    ]
    
    if len(entity_performance) > 0:
        best = entity_performance.iloc[0]
        worst = entity_performance.iloc[-1]
        lines.append(
            f"\nWithin this dataset and held-out sample:"  
        )
        lines.append(
            f"  Highest macro F1: {best['entity']} "
            f"({best['macro_f1']:.4f}, N={best['n_samples']})"
        )
        lines.append(
            f"  Lowest macro F1:  {worst['entity']} "
            f"({worst['macro_f1']:.4f}, N={worst['n_samples']})"
        )
        lines.append(
            f"\nNote: These observations are limited to this dataset's "
            f"sample and annotation. They should not be interpreted as "
            f"definitive measures of airline service quality or overall "
            f"customer sentiment."
        )
    
    return "\n".join(lines)


def save_entity_analysis(
    entity_performance: pd.DataFrame,
    entity_errors: pd.DataFrame,
    path: Optional[str] = None,
) -> None:
    """Save entity analysis artifacts."""
    perf_path = RESULTS_DIR / "entity_analysis.csv"
    entity_performance.to_csv(perf_path, index=False)
    print(f"Saved entity analysis: {perf_path}")
    
    err_path = RESULTS_DIR / "entity_error_analysis.csv"
    entity_errors.to_csv(err_path, index=False)
    print(f"Saved entity error analysis: {err_path}")
