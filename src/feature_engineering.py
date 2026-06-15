"""
Feature Engineering
===================

Creates derived features that capture financial risk signals
not directly present in the raw data.

New features:
- TotalIncome: Combined household income
- IncomeToLoan: Income-to-debt ratio (key risk indicator)
- EMI: Estimated monthly installment
- LogLoanAmount: Log-transformed loan amount (reduces right skew)
- LogTotalIncome: Log-transformed total income (reduces right skew)
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features from raw loan application data.

    This function should be called BEFORE encoding and scaling,
    on the raw (or imputed) DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with at least: ApplicantIncome, CoapplicantIncome,
        LoanAmount, Loan_Amount_Term.

    Returns
    -------
    pd.DataFrame
        DataFrame with new columns added.
    """
    df = df.copy()

    logger.info("=" * 50)
    logger.info("FEATURE ENGINEERING")
    logger.info("=" * 50)

    # ── TotalIncome ─────────────────────────
    # Combined household income — more predictive than individual incomes
    df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    logger.info(f"  TotalIncome: mean={df['TotalIncome'].mean():.0f}, "
                f"median={df['TotalIncome'].median():.0f}")

    # ── IncomeToLoan ────────────────────────
    # Higher ratio = more capacity to repay → lower risk
    df["IncomeToLoan"] = df["TotalIncome"] / (df["LoanAmount"] + 1)
    logger.info(f"  IncomeToLoan: mean={df['IncomeToLoan'].mean():.1f}, "
                f"median={df['IncomeToLoan'].median():.1f}")

    # ── EMI (Estimated Monthly Installment) ─
    # Monthly burden = LoanAmount / Term_in_months
    df["EMI"] = df["LoanAmount"] / (df["Loan_Amount_Term"] + 1)
    logger.info(f"  EMI: mean={df['EMI'].mean():.2f}, "
                f"median={df['EMI'].median():.2f}")

    # ── Log Transforms ─────────────────────
    # Reduce right-skewness in financial variables
    df["LogLoanAmount"] = np.log1p(df["LoanAmount"])
    df["LogTotalIncome"] = np.log1p(df["TotalIncome"])
    logger.info(f"  LogLoanAmount: skew reduced")
    logger.info(f"  LogTotalIncome: skew reduced")

    logger.info(f"  → Added 5 engineered features. New shape: {df.shape}")

    return df
