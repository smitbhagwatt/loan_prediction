"""
Centralized configuration for the Loan Approval Prediction pipeline.

All paths, hyperparameters, column definitions, and training settings
are managed here for reproducibility and easy experimentation.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "loan_data.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
DB_PATH = BASE_DIR / "data" / "loan_predictions.db"

# Create directories if they don't exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────
RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# ──────────────────────────────────────────────
# Dataset Schema
# ──────────────────────────────────────────────
TARGET_COL = "Loan_Status"
ID_COL = "Loan_ID"

CATEGORICAL_COLS = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
]

NUMERICAL_COLS = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
]

# Columns expected in raw data (excluding ID and target)
FEATURE_COLS = CATEGORICAL_COLS + NUMERICAL_COLS

# ──────────────────────────────────────────────
# Engineered Feature Names
# ──────────────────────────────────────────────
ENGINEERED_COLS = [
    "TotalIncome",
    "IncomeToLoan",
    "EMI",
    "LogLoanAmount",
    "LogTotalIncome",
]

# ──────────────────────────────────────────────
# Hyperparameter Grids (for GridSearchCV)
# ──────────────────────────────────────────────
LOGISTIC_REGRESSION_PARAMS = {
    "C": [0.01, 0.1, 1, 10],
    "solver": ["liblinear"],
    "max_iter": [1000],
}

DECISION_TREE_PARAMS = {
    "max_depth": [3, 5, 10, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

RANDOM_FOREST_PARAMS = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}

XGBOOST_PARAMS = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1],
    "subsample": [0.8, 1.0],
}

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ──────────────────────────────────────────────
# Streamlit App
# ──────────────────────────────────────────────
BEST_MODEL_PATH = MODEL_DIR / "best_model.joblib"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.joblib"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.joblib"
