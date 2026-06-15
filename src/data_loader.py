"""
Data Loader
===========

Loads and validates the raw loan application dataset.
Provides basic data profiling and integrity checks.
"""

import logging

import pandas as pd

from src.config import DATA_PATH, TARGET_COL, ID_COL, FEATURE_COLS

logger = logging.getLogger(__name__)


def load_data(path=None) -> pd.DataFrame:
    """
    Load the loan dataset from CSV.

    Parameters
    ----------
    path : str or Path, optional
        Override path to dataset. Defaults to config.DATA_PATH.

    Returns
    -------
    pd.DataFrame
        Raw dataset with all columns preserved.

    Raises
    ------
    FileNotFoundError
        If the dataset file does not exist.
    ValueError
        If required columns are missing from the dataset.
    """
    path = path or DATA_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            f"Run 'python data/generate_data.py' to generate it."
        )

    logger.info(f"Loading dataset from {path}")
    df = pd.read_csv(path)
    logger.info(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # ── Validate columns ────────────────────
    required_cols = set(FEATURE_COLS + [TARGET_COL, ID_COL])
    actual_cols = set(df.columns)
    missing_cols = required_cols - actual_cols

    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}. "
            f"Expected columns: {sorted(required_cols)}"
        )

    # ── Log data profile ────────────────────
    _log_profile(df)

    return df


def _log_profile(df: pd.DataFrame) -> None:
    """Log a brief data profile."""
    total_missing = df.isnull().sum().sum()
    total_cells = df.shape[0] * df.shape[1]
    missing_pct = total_missing / total_cells * 100

    logger.info(f"  Missing values: {total_missing} ({missing_pct:.1f}%)")

    # Target distribution
    target_dist = df[TARGET_COL].value_counts(normalize=True)
    for label, pct in target_dist.items():
        logger.info(f"  Target '{label}': {pct:.1%}")

    # Columns with most missing
    cols_with_missing = df.isnull().sum()
    cols_with_missing = cols_with_missing[cols_with_missing > 0].sort_values(ascending=False)
    if len(cols_with_missing) > 0:
        logger.info("  Columns with missing values:")
        for col, count in cols_with_missing.items():
            logger.info(f"    {col}: {count} ({count / len(df) * 100:.1f}%)")
