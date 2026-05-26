"""
data_pipeline.py
================
Shared data-loading and split logic used by every extension script.

Reproduces the exact training pipeline from the original notebook:
  1. Stratified 80/20 train_pool / test split   (random_state = 42)
  2. Stratified 60,000-row subsample of train_pool
  3. Stratified 75/25 train / val split on the subsample

Every extension imports `prepare_splits()` so the X_train, X_val, X_test,
y_train, y_val, y_test, amounts_test arrays are identical across scripts.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


RNG = 42

# Colour palette (used by all extension scripts)
PALETTE = {
    "fraud":   "#E24B4A",   # red
    "legit":   "#378ADD",   # blue
    "accent":  "#1D9E75",   # teal
    "neutral": "#888780",   # gray
}


class Splits(NamedTuple):
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    amounts_test: pd.Series
    feature_names: list


def _find_csv() -> Path:
    """Locate creditcard.csv — look in data/ first, then project root."""
    here = Path(__file__).resolve().parent
    candidates = [
        here.parent / "data" / "creditcard.csv",   # project_root/data/
        here.parent / "creditcard.csv",            # project_root/
        Path.cwd() / "data" / "creditcard.csv",
        Path.cwd() / "creditcard.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "creditcard.csv not found. Download it from "
        "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "
        "and place it in the data/ folder."
    )


def stratified_take(X: pd.DataFrame, y: pd.Series, n: int,
                    seed: int = RNG) -> Tuple[pd.DataFrame, pd.Series]:
    """Stratified subsample of exactly `n` rows preserving the class ratio."""
    if len(X) <= n:
        return X, y
    Xs, _, ys, _ = train_test_split(
        X, y, train_size=n, stratify=y, random_state=seed
    )
    return Xs, ys


def prepare_splits(verbose: bool = True) -> Splits:
    """
    Reproduce the exact training-pipeline splits from the notebook.

    Returns
    -------
    Splits NamedTuple with X_train/y_train (45,000 rows from 60k subsample),
    X_val/y_val (15,000 rows), X_test/y_test (~56,961 untouched test rows),
    plus amounts_test (the £ Amount column for the test set) and the
    feature_names list (V1..V28, Time, Amount).
    """
    csv_path = _find_csv()
    df = pd.read_csv(csv_path)
    y = df["Class"].astype(int)
    X = df.drop(columns=["Class"])

    # 1) Big stratified 80/20 split — test set kept hermetic until the end
    X_train_pool, X_test, y_train_pool, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RNG
    )

    # 2) Stratified 60K-row subsample of the train_pool
    X_trainsub, y_trainsub = stratified_take(X_train_pool, y_train_pool, n=60_000)

    # 3) Stratified 75/25 train/val split of the subsample
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainsub, y_trainsub, test_size=0.25,
        stratify=y_trainsub, random_state=RNG
    )

    amounts_test = X_test["Amount"].copy()

    if verbose:
        print(f"[data_pipeline] Loaded {len(df):,} rows "
              f"({y.sum():,} fraud, base rate {y.mean():.4%})")
        print(f"  Train: {len(X_train):>6,}  fraud={int(y_train.sum())}")
        print(f"  Val  : {len(X_val):>6,}  fraud={int(y_val.sum())}")
        print(f"  Test : {len(X_test):>6,}  fraud={int(y_test.sum())}")

    return Splits(
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        amounts_test=amounts_test,
        feature_names=list(X.columns),
    )


def project_root() -> Path:
    """Absolute path to the project root."""
    return Path(__file__).resolve().parent.parent


def outputs_dir() -> Path:
    """Absolute path to the outputs/ directory (creates it if missing)."""
    p = project_root() / "outputs"
    p.mkdir(exist_ok=True, parents=True)
    return p
