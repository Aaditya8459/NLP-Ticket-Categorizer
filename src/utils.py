"""Utility helpers for data loading, logging setup, and reporting."""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

import pandas as pd


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a clean format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def load_data(path: str, subject_col: str, body_col: str, label_col: str) -> pd.DataFrame:
    """
    Load ticket data from CSV, JSON, or Excel.

    Args:
        path: File path. Extension determines parser (.csv, .json, .xlsx).
        subject_col: Column name for subject.
        body_col: Column name for body.
        label_col: Column name for category label.

    Returns:
        DataFrame with exactly three columns: subject, body, category.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    ext = Path(path).suffix.lower()

    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext == ".json":
        df = pd.read_json(path)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .csv, .json, or .xlsx")

    # Validate columns
    required = {subject_col, body_col, label_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in data: {missing}. Found: {list(df.columns)}")

    # Rename to standard names
    df = df[[subject_col, body_col, label_col]].copy()
    df.columns = ["subject", "body", "category"]

    # Drop rows with missing labels
    before = len(df)
    df = df.dropna(subset=["category"])
    after = len(df)
    if after < before:
        logging.warning("Dropped %d rows with missing labels.", before - after)

    # Ensure strings
    df["subject"] = df["subject"].fillna("").astype(str)
    df["body"] = df["body"].fillna("").astype(str)
    df["category"] = df["category"].astype(str).str.strip()

    logging.info("Loaded %d tickets from %s", len(df), path)
    logging.info("Category distribution:\n%s", df["category"].value_counts().to_string())

    return df


def save_json(data: Dict[str, Any], path: str) -> None:
    """Save dict to pretty-printed JSON."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logging.info("Saved JSON report to %s", path)


def print_banner(text: str, width: int = 60) -> None:
    """Print a centered banner."""
    print("\n" + "=" * width)
    print(text.center(width))
    print("=" * width)
