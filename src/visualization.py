"""Visualization module for tweet sentiment analysis.

Creates professional, publication-quality figures for all
required analyses: EDA, model comparison, confusion matrices,
entity analysis, error analysis, and ablation studies.

All figures are saved to the outputs/figures/ directory.
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for reproducibility
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.utils import FIGURES_DIR, SENTIMENT_LABELS, SENTIMENT_CLASSES

# Consistent style
sns.set_theme(style="whitegrid", font_scale=1.1)
COLOR_PALETTE = {"negative": "#e74c3c", "neutral": "#3498db", "positive": "#2ecc71"}
MODEL_PALETTE = sns.color_palette("Set2", 8)


def _save_figure(fig: plt.Figure, filename: str, dpi: int = 150) -> Path:
    """Save figure to the figures directory."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_class_distribution(
    y: pd.Series,
    title: str = "Sentiment Class Distribution",
    filename: str = "class_distribution.png",
) -> Path:
    """Plot sentiment class distribution as a bar chart with counts and percentages."""
    counts = y.value_counts().reindex(SENTIMENT_LABELS)
    pcts = (counts / counts.sum() * 100).round(1)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [COLOR_PALETTE[label] for label in SENTIMENT_LABELS]
    bars = ax.bar(SENTIMENT_LABELS, counts.values, color=colors, edgecolor="white", linewidth=1.5)
    
    # Add count and percentage labels
    for bar, count, pct in zip(bars, counts.values, pcts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + counts.max() * 0.02,
            f"{count:,}\n({pct}%)",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )
    
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Sentiment Class", fontsize=12)
    ax.set_ylabel("Number of Tweets", fontsize=12)
    ax.set_ylim(0, counts.max() * 1.2)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    sns.despine()
    
    return _save_figure(fig, filename)


def plot_tweet_length_distribution(
    df: pd.DataFrame,
    text_col: str = "text",
    target_col: str = "airline_sentiment",
    filename: str = "tweet_length_distribution.png",
) -> Path:
    """Plot tweet character length distribution by sentiment class."""
    df = df.copy()
    df["char_length"] = df[text_col].str.len()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram by sentiment
    for label in SENTIMENT_LABELS:
        mask = df[target_col] == label
        axes[0].hist(
            df.loc[mask, "char_length"],
            bins=50, alpha=0.6, label=label,
            color=COLOR_PALETTE[label], edgecolor="white",
        )
    axes[0].set_title("Tweet Character Length by Sentiment", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Character Length", fontsize=11)
    axes[0].set_ylabel("Count", fontsize=11)
    axes[0].legend(title="Sentiment")
    
    # Box plot
    order = SENTIMENT_LABELS
    colors = [COLOR_PALETTE[l] for l in order]
    bp = sns.boxplot(
        data=df, x=target_col, y="char_length",
        order=order, palette=COLOR_PALETTE, ax=axes[1],
        fliersize=2,
    )
    axes[1].set_title("Tweet Length Distribution by Sentiment", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Sentiment Class", fontsize=11)
    axes[1].set_ylabel("Character Length", fontsize=11)
    
    fig.tight_layout()
    return _save_figure(fig, filename)


def plot_missing_values(
    df: pd.DataFrame,
    filename: str = "missing_values.png",
) -> Path:
    """Plot missing values per column."""
    null_counts = df.isnull().sum()
    null_pcts = (null_counts / len(df) * 100).round(2)
    
    # Only show columns with missing values, plus a few key ones
    has_missing = null_counts[null_counts > 0]
    if len(has_missing) == 0:
        # Show all columns to demonstrate no missing data
        has_missing = null_counts
    
    fig, ax = plt.subplots(figsize=(10, max(5, len(has_missing) * 0.4)))
    
    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in has_missing.values]
    bars = ax.barh(has_missing.index, has_missing.values, color=colors, edgecolor="white")
    
    for bar, count, pct in zip(bars, has_missing.values, null_pcts[has_missing.index].values):
        if count > 0:
            ax.text(
                bar.get_width() + has_missing.max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{count:,} ({pct}%)",
                ha="left", va="center", fontsize=9,
            )
    
    ax.set_title("Missing Values by Column", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Missing Values", fontsize=12)
    ax.invert_yaxis()
    sns.despine()
    
    return _save_figure(fig, filename)


def plot_top_terms_by_class(
    tfidf_vectorizer,
    X_tfidf,
    y_train: pd.Series,
    n_terms: int = 15,
    filename: str = "top_terms_by_class.png",
) -> Path:
    """Plot top TF-IDF terms per sentiment class (training data only)."""
    feature_names = tfidf_vectorizer.get_feature_names_out()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, (label, ax) in enumerate(zip(SENTIMENT_LABELS, axes)):
        mask = y_train.values == label
        if hasattr(X_tfidf, 'toarray'):
            class_tfidf = X_tfidf[mask].mean(axis=0).A1
        else:
            class_tfidf = X_tfidf[mask].mean(axis=0)
        
        top_indices = class_tfidf.argsort()[-n_terms:][::-1]
        top_terms = [feature_names[i] for i in top_indices]
        top_scores = [class_tfidf[i] for i in top_indices]
        
        color = COLOR_PALETTE[label]
        ax.barh(range(n_terms), top_scores[::-1], color=color, edgecolor="white")
        ax.set_yticks(range(n_terms))
        ax.set_yticklabels(top_terms[::-1], fontsize=9)
        ax.set_title(f"Top Terms: {label.capitalize()}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Mean TF-IDF Score", fontsize=10)
    
    fig.suptitle("Top TF-IDF Terms by Sentiment Class (Training Data Only)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save_figure(fig, filename)


def plot_model_comparison(
    cv_df: pd.DataFrame,
    metric: str = "mean_macro_f1",
    std_col: str = "std_macro_f1",
    filename: str = "model_comparison.png",
) -> Path:
    """Plot model comparison from cross-validation results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Macro F1 comparison with error bars
    models = cv_df["model"].values
    means = cv_df[metric].values
    stds = cv_df[std_col].values
    colors = MODEL_PALETTE[:len(models)]
    
    bars = axes[0].bar(models, means, yerr=stds, color=colors,
                       edgecolor="white", linewidth=1.5, capsize=5)
    for bar, mean, std in zip(bars, means, stds):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.005,
            f"{mean:.4f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )
    axes[0].set_title("Cross-Validation: Macro F1", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Macro F1 Score", fontsize=11)
    axes[0].set_ylim(0, min(1.0, max(means) + max(stds) + 0.1))
    
    # Multi-metric comparison
    metrics_to_plot = ["mean_accuracy", "mean_macro_f1", "mean_weighted_f1",
                       "mean_macro_precision", "mean_macro_recall"]
    metric_labels = ["Accuracy", "Macro F1", "Weighted F1", "Macro Prec.", "Macro Recall"]
    
    x = np.arange(len(metrics_to_plot))
    width = 0.8 / len(models)
    
    for i, (model, color) in enumerate(zip(models, colors)):
        row = cv_df[cv_df["model"] == model].iloc[0]
        values = [row[m] for m in metrics_to_plot]
        axes[1].bar(x + i * width, values, width, label=model, color=color, edgecolor="white")
    
    axes[1].set_xticks(x + width * (len(models) - 1) / 2)
    axes[1].set_xticklabels(metric_labels, fontsize=9)
    axes[1].set_title("Multi-Metric Comparison", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Score", fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].set_ylim(0, 1.05)
    
    fig.tight_layout()
    return _save_figure(fig, filename)


def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    labels: List[str] = SENTIMENT_LABELS,
    normalize: bool = False,
    title: str = "Confusion Matrix",
    filename: str = "confusion_matrix_count.png",
) -> Path:
    """Plot a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    if normalize:
        cm_plot = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2%"
        title = title + " (Row-Normalized)"
    else:
        cm_plot = cm
        fmt = "d"
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_plot, annot=True, fmt=fmt, cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        ax=ax, linewidths=0.5, linecolor="white",
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.tick_params(axis="both", labelsize=10)
    
    return _save_figure(fig, filename)


def plot_per_class_metrics(
    classification_report: Dict,
    model_name: str = "Selected Model",
    filename: str = "per_class_metrics.png",
) -> Path:
    """Plot per-class precision, recall, and F1 as grouped bars."""
    metrics = ["precision", "recall", "f1-score"]
    labels = SENTIMENT_LABELS
    
    data = []
    for label in labels:
        for metric in metrics:
            data.append({
                "Class": label.capitalize(),
                "Metric": metric.replace("f1-score", "F1").capitalize(),
                "Value": classification_report[label][metric],
            })
    
    plot_df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=plot_df, x="Class", y="Value", hue="Metric",
        palette=["#3498db", "#e74c3c", "#2ecc71"], ax=ax,
        edgecolor="white", linewidth=1.5,
    )
    
    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=9, padding=3)
    
    ax.set_title(f"Per-Class Metrics: {model_name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Sentiment Class", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.legend(title="Metric", fontsize=10)
    sns.despine()
    
    return _save_figure(fig, filename)


def plot_entity_sentiment_distribution(
    df: pd.DataFrame,
    entity_col: str = "airline",
    target_col: str = "airline_sentiment",
    filename: str = "entity_sentiment_distribution.png",
) -> Path:
    """Plot sentiment distribution per airline/entity."""
    ct = pd.crosstab(df[entity_col], df[target_col], normalize="index") * 100
    ct = ct.reindex(columns=SENTIMENT_LABELS)
    
    # Sort by negative proportion
    ct = ct.sort_values("negative", ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ct.plot(
        kind="barh", stacked=True, ax=ax,
        color=[COLOR_PALETTE[l] for l in SENTIMENT_LABELS],
        edgecolor="white", linewidth=0.5,
    )
    
    # Add count annotations
    entity_counts = df[entity_col].value_counts()
    for i, entity in enumerate(ct.index):
        ax.text(
            102, i, f"N={entity_counts[entity]:,}",
            va="center", fontsize=9, style="italic",
        )
    
    ax.set_title("Sentiment Distribution by Airline", fontsize=14, fontweight="bold")
    ax.set_xlabel("Percentage (%)", fontsize=12)
    ax.set_ylabel("")
    ax.legend(
        title="Sentiment", bbox_to_anchor=(1.15, 1), loc="upper left",
        fontsize=10,
    )
    ax.set_xlim(0, 115)
    
    return _save_figure(fig, filename)


def plot_entity_error_rate(
    entity_perf: pd.DataFrame,
    filename: str = "entity_error_rate.png",
) -> Path:
    """Plot error rate per entity with sample size annotations."""
    if len(entity_perf) == 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "No entities with sufficient support",
                ha="center", va="center", fontsize=14)
        return _save_figure(fig, filename)
    
    df = entity_perf.sort_values("error_rate", ascending=True).copy()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(
        df["entity"], df["error_rate"],
        color=sns.color_palette("YlOrRd", len(df)),
        edgecolor="white", linewidth=1.5,
    )
    
    for bar, (_, row) in zip(bars, df.iterrows()):
        ax.text(
            bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{row['error_rate']:.1%} (N={row['n_samples']})",
            va="center", fontsize=10,
        )
    
    ax.set_title("Classifier Error Rate by Airline (Locked Test Set)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Error Rate", fontsize=12)
    ax.set_xlim(0, df["error_rate"].max() * 1.4)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    sns.despine()
    
    return _save_figure(fig, filename)


def plot_preprocessing_ablation(
    ablation_df: pd.DataFrame,
    filename: str = "preprocessing_ablation.png",
) -> Path:
    """Plot preprocessing ablation comparison."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    metrics = ["macro_f1", "weighted_f1", "accuracy"]
    metric_labels = ["Macro F1", "Weighted F1", "Accuracy"]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    strategies = ablation_df["strategy"].unique()
    colors = ["#3498db", "#e74c3c"]
    
    for i, strategy in enumerate(strategies):
        row = ablation_df[ablation_df["strategy"] == strategy].iloc[0]
        values = [row[m] for m in metrics]
        bars = ax.bar(x + i * width, values, width, label=strategy.capitalize(),
                      color=colors[i], edgecolor="white", linewidth=1.5)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.4f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
            )
    
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_title("Preprocessing Ablation: Minimal vs. Aggressive",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.legend(title="Strategy", fontsize=10)
    sns.despine()
    
    return _save_figure(fig, filename)


def plot_confidence_error(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    filename: str = "confidence_error.png",
) -> Path:
    """Plot relationship between prediction confidence and error rate."""
    max_proba = y_proba.max(axis=1)
    correct = y_true.values == y_pred
    
    # Bin by confidence
    bins = np.arange(0.3, 1.05, 0.1)
    bin_labels = [f"{b:.1f}-{b+0.1:.1f}" for b in bins[:-1]]
    binned = np.digitize(max_proba, bins)
    
    error_rates = []
    counts = []
    for i in range(1, len(bins)):
        mask = binned == i
        if mask.sum() > 0:
            error_rates.append(1 - correct[mask].mean())
            counts.append(mask.sum())
        else:
            error_rates.append(0)
            counts.append(0)
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    ax1.bar(bin_labels, counts, color="#3498db", alpha=0.5, label="Sample Count")
    ax1.set_xlabel("Confidence Bin", fontsize=12)
    ax1.set_ylabel("Sample Count", fontsize=12, color="#3498db")
    
    ax2 = ax1.twinx()
    ax2.plot(bin_labels, error_rates, "ro-", linewidth=2, markersize=8, label="Error Rate")
    ax2.set_ylabel("Error Rate", fontsize=12, color="red")
    ax2.set_ylim(0, 1)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    
    ax1.set_title("Prediction Confidence vs. Error Rate",
                  fontsize=14, fontweight="bold")
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center")
    
    fig.tight_layout()
    return _save_figure(fig, filename)
