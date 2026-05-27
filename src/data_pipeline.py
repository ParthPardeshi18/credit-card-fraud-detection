"""
data_pipeline.py
================
Shared data-loading, remote-fetch and split logic used by every script.

Dataset loading order
---------------------
1. data/creditcard.parquet  (local dev — fastest, 46 MB, ignored by git)
2. data/creditcard.csv      (local legacy — 144 MB, also git-ignored)
3. HuggingFace Hub          (Streamlit Cloud / CI — downloads once, ~46 MB)
4. FileNotFoundError        (with clear setup instructions)

Streamlit Cloud integration
---------------------------
The module exposes ``st_load_dataframe()`` which wraps the loader in
``@st.cache_data`` so Streamlit re-uses the DataFrame across reruns and
only re-downloads when the TTL expires.

For standalone scripts (setup_model.py, shap_analysis.py etc.) call
``load_dataframe()`` directly — no caching overhead.

Training split reproducibility
--------------------------------
Reproduces the exact pipeline from the original QMUL notebook:
  1. Stratified 80/20 train_pool / test    (random_state = 42)
  2. Stratified 60,000-row subsample of train_pool
  3. Stratified 75/25 train / val on the subsample
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RNG = 42

# HuggingFace Hub — update DATASET_REPO to your own HF dataset after upload.
# Upload instructions:   docs/DATASET_HOSTING.md
# Quick CLI upload:      huggingface-cli upload <user>/credit-card-fraud data/creditcard.parquet
HF_DATASET_REPO = "ParthPardeshi18/credit-card-fraud"          # <── change to your HF repo
HF_PARQUET_FILE = "creditcard.parquet"
HF_BASE_URL     = "https://huggingface.co/datasets"
PARQUET_URL     = f"{HF_BASE_URL}/{HF_DATASET_REPO}/resolve/main/{HF_PARQUET_FILE}"

# Disk-cache location for downloaded data (avoids re-download every session)
_CACHE_DIR   = Path(tempfile.gettempdir()) / "qmul_fraud_data"
_CACHE_FILE  = _CACHE_DIR / "creditcard.parquet"
_CACHE_TTL_S = 86_400   # re-download after 24 h (set 0 to always re-download)

# Colour palette (used by all extension scripts)
PALETTE = {
    "fraud":   "#E24B4A",
    "legit":   "#378ADD",
    "accent":  "#1D9E75",
    "neutral": "#888780",
}


# ---------------------------------------------------------------------------
# Named tuple for split outputs
# ---------------------------------------------------------------------------

class Splits(NamedTuple):
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val:   pd.DataFrame
    y_val:   pd.Series
    X_test:  pd.DataFrame
    y_test:  pd.Series
    amounts_test: pd.Series
    feature_names: list


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def project_root() -> Path:
    """Absolute path to the project root (parent of src/)."""
    return Path(__file__).resolve().parent.parent


def outputs_dir() -> Path:
    """Absolute path to outputs/ (creates it if missing)."""
    p = project_root() / "outputs"
    p.mkdir(exist_ok=True, parents=True)
    return p


# ---------------------------------------------------------------------------
# Data loading — local → remote chain
# ---------------------------------------------------------------------------

def _local_candidates() -> list[Path]:
    """Return ordered list of local file candidates (fastest first)."""
    root = project_root()
    return [
        root / "data" / "creditcard.parquet",
        root / "data" / "creditcard.csv",
        Path.cwd() / "data" / "creditcard.parquet",
        Path.cwd() / "data" / "creditcard.csv",
        Path.cwd() / "creditcard.parquet",
        Path.cwd() / "creditcard.csv",
    ]


def _read_file(path: Path) -> pd.DataFrame:
    """Read a parquet or CSV file into a DataFrame."""
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _is_cache_fresh() -> bool:
    """True if the disk-cache exists and was written within TTL."""
    if not _CACHE_FILE.exists():
        return False
    if _CACHE_TTL_S == 0:
        return False
    age = time.time() - _CACHE_FILE.stat().st_mtime
    return age < _CACHE_TTL_S


def _download_from_hf(verbose: bool = True) -> pd.DataFrame:
    """
    Download the parquet from HuggingFace Hub.

    Uses a temp-file + atomic rename so a failed/partial download
    never leaves a corrupt cache file.
    """
    try:
        import requests
    except ImportError as exc:
        raise ImportError(
            "The 'requests' package is required for remote loading. "
            "Install it with:  pip install requests"
        ) from exc

    if verbose:
        print(f"[data_pipeline] Downloading dataset from HuggingFace Hub …")
        print(f"  URL: {PARQUET_URL}")

    # Optional HF token for private repos (set HF_TOKEN env var or
    # store in .streamlit/secrets.toml under [secrets] HF_TOKEN = "hf_...")
    token = os.environ.get("HF_TOKEN") or _read_streamlit_secret("HF_TOKEN")
    headers: dict = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    response = requests.get(PARQUET_URL, headers=headers, stream=True, timeout=120)

    if response.status_code == 404:
        raise FileNotFoundError(
            f"Dataset not found at {PARQUET_URL}\n\n"
            "Set up HuggingFace hosting by following docs/DATASET_HOSTING.md\n"
            "or place creditcard.parquet / creditcard.csv in the data/ folder."
        )
    response.raise_for_status()

    # Stream to a temp file then atomically rename to cache
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = _CACHE_FILE.with_suffix(".tmp")

    total = int(response.headers.get("content-length", 0))
    downloaded = 0
    with open(tmp_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1 << 20):   # 1 MB chunks
            f.write(chunk)
            downloaded += len(chunk)
            if verbose and total:
                pct = downloaded / total * 100
                print(f"\r  {downloaded/1e6:.1f}/{total/1e6:.1f} MB  ({pct:.0f}%)",
                      end="", flush=True)

    tmp_path.rename(_CACHE_FILE)

    if verbose:
        print(f"\n  Cached to: {_CACHE_FILE}")

    return pd.read_parquet(_CACHE_FILE)


def _read_streamlit_secret(key: str) -> Optional[str]:
    """Silently retrieve a Streamlit secret (returns None if not available)."""
    try:
        import streamlit as st
        return st.secrets.get(key)
    except Exception:
        return None


def load_dataframe(verbose: bool = True) -> pd.DataFrame:
    """
    Load the credit-card dataset as a DataFrame.

    Loading order:
      1. Local parquet / CSV in data/  (dev workflow, git-ignored)
      2. Disk cache from previous download  (~/.../qmul_fraud_data/)
      3. HuggingFace Hub download + cache to disk
      4. Raise FileNotFoundError with setup instructions

    Parameters
    ----------
    verbose : bool
        Print progress to stdout (disabled inside Streamlit via st_load_dataframe).

    Returns
    -------
    pd.DataFrame  with columns Time, V1..V28, Amount, Class
    """
    # 1 ─ Local file (fastest, for development)
    for p in _local_candidates():
        if p.exists():
            if verbose:
                print(f"[data_pipeline] Loading from local: {p}")
            df = _read_file(p)
            if verbose:
                _print_summary(df)
            return df

    # 2 ─ Disk cache from a previous download
    if _is_cache_fresh():
        if verbose:
            print(f"[data_pipeline] Loading from disk cache: {_CACHE_FILE}")
        df = pd.read_parquet(_CACHE_FILE)
        if verbose:
            _print_summary(df)
        return df

    # 3 ─ Remote download from HuggingFace Hub
    df = _download_from_hf(verbose=verbose)
    if verbose:
        _print_summary(df)
    return df


def _print_summary(df: pd.DataFrame) -> None:
    n_fraud = int(df["Class"].sum())
    print(f"[data_pipeline] Loaded {len(df):,} rows "
          f"({n_fraud:,} fraud, base rate {df['Class'].mean():.4%})")


# ---------------------------------------------------------------------------
# Streamlit-cached loader (use this inside fraud_app.py)
# ---------------------------------------------------------------------------

def st_load_dataframe() -> pd.DataFrame:
    """
    ``@st.cache_data``-wrapped version of ``load_dataframe()``.

    Import and call this inside fraud_app.py so the DataFrame is loaded
    once per Streamlit session and shared across all reruns.

    Usage::

        from data_pipeline import st_load_dataframe
        df = st_load_dataframe()     # downloads once, then cached
    """
    try:
        import streamlit as st
    except ImportError as exc:
        raise ImportError("streamlit is not installed.") from exc

    @st.cache_data(show_spinner="⏳  Loading dataset …", ttl=_CACHE_TTL_S)
    def _cached() -> pd.DataFrame:
        return load_dataframe(verbose=False)

    return _cached()


# ---------------------------------------------------------------------------
# Stratified helpers & split logic
# ---------------------------------------------------------------------------

def stratified_take(X: pd.DataFrame, y: pd.Series, n: int,
                    seed: int = RNG) -> Tuple[pd.DataFrame, pd.Series]:
    """Stratified subsample of exactly *n* rows (preserves class ratio)."""
    if len(X) <= n:
        return X, y
    Xs, _, ys, _ = train_test_split(
        X, y, train_size=n, stratify=y, random_state=seed
    )
    return Xs, ys


def prepare_splits(verbose: bool = True, df: Optional[pd.DataFrame] = None) -> Splits:
    """
    Reproduce the exact training-pipeline splits from the QMUL notebook.

    Parameters
    ----------
    verbose : bool
        Print split sizes to stdout.
    df : pd.DataFrame, optional
        Pre-loaded DataFrame.  If None, ``load_dataframe()`` is called.
        Pass a cached df from ``st_load_dataframe()`` inside the Streamlit
        app to avoid redundant I/O.

    Returns
    -------
    Splits  NamedTuple with X_train / y_train (45 k), X_val / y_val (15 k),
            X_test / y_test (~57 k), amounts_test, feature_names.
    """
    if df is None:
        df = load_dataframe(verbose=verbose)

    y = df["Class"].astype(int)
    X = df.drop(columns=["Class"])

    # 1) 80/20 hermetic test split
    X_pool, X_test, y_pool, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RNG
    )

    # 2) Stratified 60 k subsample of train_pool
    X_sub, y_sub = stratified_take(X_pool, y_pool, n=60_000)

    # 3) 75/25 train / val
    X_train, X_val, y_train, y_val = train_test_split(
        X_sub, y_sub, test_size=0.25, stratify=y_sub, random_state=RNG
    )

    amounts_test = X_test["Amount"].copy()

    if verbose:
        print(f"  Train: {len(X_train):>6,}  fraud={int(y_train.sum())}")
        print(f"  Val  : {len(X_val):>6,}  fraud={int(y_val.sum())}")
        print(f"  Test : {len(X_test):>6,}  fraud={int(y_test.sum())}")

    return Splits(
        X_train=X_train, y_train=y_train,
        X_val=X_val,     y_val=y_val,
        X_test=X_test,   y_test=y_test,
        amounts_test=amounts_test,
        feature_names=list(X.columns),
    )
