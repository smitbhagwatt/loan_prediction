"""
Data Preprocessing
==================

Handles missing values, encodes categorical features, and scales
numerical features. Returns processed data ready for model training.

All fitted transformers are saved for reproducible inference.
"""

import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import (
    CATEGORICAL_COLS,
    NUMERICAL_COLS,
    TARGET_COL,
    ID_COL,
    TEST_SIZE,
    RANDOM_SEED,
    PREPROCESSOR_PATH,
)

logger = logging.getLogger(__name__)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values using domain-appropriate strategies.

    - Numerical columns → median (robust to outliers)
    - Categorical columns → mode (most frequent value)
    - Credit_History → mode (treated as categorical despite being numeric)
    """
    df = df.copy()
    total_before = df.isnull().sum().sum()

    # Categorical imputation (mode)
    for col in CATEGORICAL_COLS:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0]
            count = df[col].isnull().sum()
            df[col] = df[col].fillna(mode_val)
            logger.info(f"  Imputed {col}: {count} missing -> '{mode_val}' (mode)")

    # Numerical imputation (median)
    for col in NUMERICAL_COLS:
        if df[col].isnull().any():
            median_val = df[col].median()
            count = df[col].isnull().sum()
            df[col] = df[col].fillna(median_val)
            logger.info(f"  Imputed {col}: {count} missing -> {median_val:.1f} (median)")

    total_after = df.isnull().sum().sum()
    logger.info(f"  Missing values: {total_before} -> {total_after}")

    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Encode target variable: Y → 1, N → 0."""
    df = df.copy()
    df[TARGET_COL] = df[TARGET_COL].map({"Y": 1, "N": 0})
    logger.info(f"  Encoded target: Y→1, N→0")
    return df


def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Encode categorical features using LabelEncoder.

    Returns the encoded DataFrame and a dict of fitted encoders
    for use during inference.
    """
    df = df.copy()
    encoders = {}

    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        logger.info(f"  Encoded {col}: {list(le.classes_)} → {list(range(len(le.classes_)))}")

    return df, encoders


def scale_numerical(df: pd.DataFrame, columns: list) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Standardize numerical features (zero mean, unit variance).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with numerical columns to scale.
    columns : list
        Column names to scale.

    Returns
    -------
    tuple[pd.DataFrame, StandardScaler]
        Scaled DataFrame and fitted scaler.
    """
    df = df.copy()
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    logger.info(f"  Scaled {len(columns)} numerical columns (StandardScaler)")
    return df, scaler


def preprocess(df: pd.DataFrame) -> dict:
    """
    Full preprocessing pipeline.

    Steps:
    1. Drop Loan_ID
    2. Impute missing values
    3. Encode target (Y/N → 1/0)
    4. Encode categorical features
    5. Split into train/test
    6. Scale numerical features (fit on train only)
    7. Save preprocessing artifacts

    Returns
    -------
    dict with keys:
        X_train, X_test, y_train, y_test,
        encoders, scaler, feature_names
    """
    logger.info("=" * 50)
    logger.info("PREPROCESSING")
    logger.info("=" * 50)

    # Drop ID column
    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])
        logger.info(f"  Dropped '{ID_COL}' column")

    # Step 1: Handle missing values (BEFORE feature engineering)
    logger.info("Step 1: Handling missing values")
    df = handle_missing_values(df)

    # Step 2: Feature engineering (on imputed data — no NaN propagation)
    logger.info("Step 2: Feature engineering")
    from src.feature_engineering import engineer_features
    df = engineer_features(df)

    # Step 3: Encode target
    logger.info("Step 3: Encoding target variable")
    df = encode_target(df)

    # Step 4: Encode categoricals
    logger.info("Step 4: Encoding categorical features")
    df, encoders = encode_categoricals(df)

    # Step 5: Train/test split (BEFORE scaling to prevent data leakage)
    logger.info("Step 5: Train/test split")
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    logger.info(f"  Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
    logger.info(f"  Train approval rate: {y_train.mean():.1%} | Test: {y_test.mean():.1%}")

    # Step 6: Scale numerical features (fit on train only)
    logger.info("Step 6: Scaling numerical features")
    # Scale all numerical columns including engineered ones
    from src.config import ENGINEERED_COLS
    all_numerical = NUMERICAL_COLS + [c for c in ENGINEERED_COLS if c in X_train.columns]
    cols_to_scale = [c for c in all_numerical if c in X_train.columns]

    scaler = StandardScaler()
    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    feature_names = list(X_train.columns)
    logger.info(f"  Final features ({len(feature_names)}): {feature_names}")

    # Step 7: Save preprocessing artifacts
    logger.info("Step 7: Saving preprocessing artifacts")
    artifacts = {
        "encoders": encoders,
        "scaler": scaler,
        "feature_names": feature_names,
    }
    joblib.dump(artifacts, PREPROCESSOR_PATH)
    logger.info(f"  Saved preprocessor to {PREPROCESSOR_PATH}")

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "encoders": encoders,
        "scaler": scaler,
        "feature_names": feature_names,
    }
