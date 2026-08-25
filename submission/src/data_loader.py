"""Dataset loading, schema validation, governance, and splitting.

Handles the Twitter US Airline Sentiment dataset lifecycle:
loading, inspection, validation, governance audit, and
leakage-safe stratified splitting.
"""

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import (
    SEED, TEST_SIZE, VAL_SIZE, DATA_DIR, ARTIFACTS_DIR, RESULTS_DIR,
    DATASET_FILENAME, TEXT_COLUMN, TARGET_COLUMN, ENTITY_COLUMN,
    TWEET_ID_COLUMN, SENTIMENT_CLASSES, LEAKAGE_COLUMNS,
    PRIVACY_COLUMNS, EXCLUDED_COLUMNS, save_json, print_section,
)


def load_dataset(filepath: Optional[Path] = None) -> pd.DataFrame:
    """Load the Twitter US Airline Sentiment dataset.
    
    Parameters
    ----------
    filepath : Path, optional
        Path to the CSV file. Defaults to DATA_DIR / DATASET_FILENAME.
    
    Returns
    -------
    pd.DataFrame
        The loaded dataset.
    
    Raises
    ------
    FileNotFoundError
        If the dataset file is not found.
    """
    if filepath is None:
        filepath = DATA_DIR / DATASET_FILENAME
    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found at {filepath}. "
            f"Please download from https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment "
            f"and place '{DATASET_FILENAME}' in {DATA_DIR}/"
        )
    df = pd.read_csv(filepath)
    print(f"Loaded dataset: {filepath}")
    print(f"  Shape: {df.shape}")
    return df


def inspect_schema(df: pd.DataFrame) -> Dict:
    """Inspect and document the dataset schema.
    
    Returns a dict containing column names, dtypes, sample values,
    null counts, and detected roles (text, target, entity, id, metadata).
    """
    schema = {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "columns": {},
    }
    for col in df.columns:
        col_info = {
            "dtype": str(df[col].dtype),
            "null_count": int(df[col].isnull().sum()),
            "null_pct": round(df[col].isnull().mean() * 100, 2),
            "n_unique": int(df[col].nunique()),
            "sample_values": df[col].dropna().head(3).tolist(),
        }
        # Identify column roles
        if col == TEXT_COLUMN:
            col_info["role"] = "text_input"
        elif col == TARGET_COLUMN:
            col_info["role"] = "target"
        elif col == ENTITY_COLUMN:
            col_info["role"] = "entity"
        elif col == TWEET_ID_COLUMN:
            col_info["role"] = "identifier"
        elif col in LEAKAGE_COLUMNS:
            col_info["role"] = "leakage_risk"
        elif col in PRIVACY_COLUMNS:
            col_info["role"] = "privacy_sensitive"
        else:
            col_info["role"] = "metadata"
        schema["columns"][col] = col_info
    return schema


def validate_dataset(df: pd.DataFrame) -> Dict[str, bool]:
    """Validate critical dataset properties.
    
    Returns a dict of validation checks and their pass/fail status.
    """
    checks = {}
    # Text column exists
    checks["text_column_exists"] = TEXT_COLUMN in df.columns
    # Target column exists
    checks["target_column_exists"] = TARGET_COLUMN in df.columns
    # Target has expected classes
    if checks["target_column_exists"]:
        actual_classes = sorted(df[TARGET_COLUMN].dropna().unique().tolist())
        checks["target_classes_valid"] = actual_classes == sorted(SENTIMENT_CLASSES)
        checks["target_no_nulls"] = df[TARGET_COLUMN].isnull().sum() == 0
    # Entity column exists
    checks["entity_column_exists"] = ENTITY_COLUMN in df.columns
    # Text has no nulls
    checks["text_no_nulls"] = df[TEXT_COLUMN].isnull().sum() == 0 if TEXT_COLUMN in df.columns else False
    # Print results
    print("\nDataset Validation:")
    for check, passed in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {check}")
    return checks


def analyze_duplicates(df: pd.DataFrame) -> Dict:
    """Analyze duplicate tweets by ID and text content."""
    result = {}
    # Duplicate tweet IDs
    if TWEET_ID_COLUMN in df.columns:
        dup_ids = df[TWEET_ID_COLUMN].duplicated().sum()
        result["duplicate_tweet_ids"] = int(dup_ids)
    # Duplicate text
    dup_text = df[TEXT_COLUMN].duplicated().sum()
    result["duplicate_text"] = int(dup_text)
    result["duplicate_text_pct"] = round(dup_text / len(df) * 100, 2)
    # Show examples of duplicated text
    if dup_text > 0:
        dup_mask = df[TEXT_COLUMN].duplicated(keep=False)
        dup_examples = df[dup_mask].groupby(TEXT_COLUMN).size().sort_values(ascending=False)
        result["top_duplicate_texts"] = dup_examples.head(5).to_dict()
    return result


def governance_audit(df: pd.DataFrame) -> Dict:
    """Perform a comprehensive data governance audit.
    
    Checks for leakage risks, privacy concerns, and documents
    all column roles and exclusion decisions.
    """
    audit = {
        "total_columns": len(df.columns),
        "leakage_columns_found": [],
        "privacy_columns_found": [],
        "modeling_columns": [],
        "excluded_columns": [],
    }
    for col in df.columns:
        if col in LEAKAGE_COLUMNS:
            audit["leakage_columns_found"].append(col)
            audit["excluded_columns"].append(col)
        elif col in PRIVACY_COLUMNS:
            audit["privacy_columns_found"].append(col)
            audit["excluded_columns"].append(col)
        elif col in [TEXT_COLUMN, TARGET_COLUMN, ENTITY_COLUMN]:
            audit["modeling_columns"].append(col)
    audit["leakage_columns_detected"] = len(audit["leakage_columns_found"])
    audit["privacy_columns_detected"] = len(audit["privacy_columns_found"])
    return audit


def create_stratified_split(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    seed: int = SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create a leakage-safe stratified train/val/test split.
    
    Parameters
    ----------
    df : pd.DataFrame
        Complete dataset.
    test_size : float
        Proportion for the locked test set.
    val_size : float
        Proportion for validation (from remaining after test).
    seed : int
        Random state for reproducibility.
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train_df, val_df, test_df) with disjoint indices.
    """
    # First split: train+val vs test
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed,
        stratify=df[TARGET_COLUMN]
    )
    # Second split: train vs val (val_size relative to original)
    # Adjust val proportion relative to train_val
    val_relative = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_relative, random_state=seed,
        stratify=train_val_df[TARGET_COLUMN]
    )
    print(f"\nSplit Summary (seed={seed}):")
    print(f"  Train: {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(val_df):,} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)")
    # Verify disjointness
    assert len(set(train_df.index) & set(test_df.index)) == 0, "Train/test overlap!"
    assert len(set(train_df.index) & set(val_df.index)) == 0, "Train/val overlap!"
    assert len(set(val_df.index) & set(test_df.index)) == 0, "Val/test overlap!"
    return train_df, val_df, test_df


def save_split_manifest(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int = SEED,
) -> pd.DataFrame:
    """Save split assignments to a manifest CSV."""
    manifests = []
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        manifest = pd.DataFrame({
            "index": split_df.index,
            "split": split_name,
        })
        if TWEET_ID_COLUMN in split_df.columns:
            manifest["tweet_id"] = split_df[TWEET_ID_COLUMN].values
        manifests.append(manifest)
    manifest_df = pd.concat(manifests, ignore_index=True)
    manifest_path = ARTIFACTS_DIR / "split_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    print(f"Saved split manifest: {manifest_path}")
    
    # Also save summary
    summary = {
        "seed": seed,
        "strategy": "stratified",
        "test_size": TEST_SIZE,
        "val_size": VAL_SIZE,
        "train_count": len(train_df),
        "val_count": len(val_df),
        "test_count": len(test_df),
        "train_distribution": train_df[TARGET_COLUMN].value_counts().to_dict(),
        "val_distribution": val_df[TARGET_COLUMN].value_counts().to_dict(),
        "test_distribution": test_df[TARGET_COLUMN].value_counts().to_dict(),
    }
    save_json(summary, ARTIFACTS_DIR / "split_summary.json")
    return manifest_df


def create_dataset_manifest(df: pd.DataFrame, schema: Dict, duplicates: Dict) -> Dict:
    """Create and save a comprehensive dataset manifest."""
    manifest = {
        "dataset_name": "Twitter US Airline Sentiment",
        "source": "CrowdFlower / Kaggle",
        "url": "https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment",
        "access_date": "2026-08-24",
        "license": "CC BY-NC-SA 4.0 (CrowdFlower open data)",
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "columns": list(df.columns),
        "target_column": TARGET_COLUMN,
        "text_column": TEXT_COLUMN,
        "entity_column": ENTITY_COLUMN,
        "sentiment_classes": SENTIMENT_CLASSES,
        "class_distribution": df[TARGET_COLUMN].value_counts().to_dict(),
        "class_percentages": (df[TARGET_COLUMN].value_counts(normalize=True) * 100).round(2).to_dict(),
        "entity_list": sorted(df[ENTITY_COLUMN].unique().tolist()) if ENTITY_COLUMN in df.columns else [],
        "language": "English (Twitter/social media)",
        "domain": "US airline customer tweets",
        "collection_period": "February 2015",
        "annotation_method": "CrowdFlower human annotators",
        "duplicates": duplicates,
        "missing_values": df.isnull().sum().to_dict(),
        "schema": schema,
        "file_hash_md5": None,  # Computed at load time if file exists
    }
    # Try to compute file hash
    filepath = DATA_DIR / DATASET_FILENAME
    if filepath.exists():
        with open(filepath, "rb") as f:
            manifest["file_hash_md5"] = hashlib.md5(f.read()).hexdigest()
    
    save_json(manifest, ARTIFACTS_DIR / "dataset_manifest.json")
    return manifest
