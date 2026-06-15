"""
Synthetic Loan Application Dataset Generator
=============================================

Generates a realistic loan application dataset with:
- Proper demographic distributions
- Correlated financial features
- Realistic missing value patterns
- Target variable driven by interpretable rules

Usage:
    python data/generate_data.py              # Default: 8000 rows
    python data/generate_data.py --n 10000    # Custom size
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────
# Constants — Realistic Distributions
# ──────────────────────────────────────────────

GENDER_PROBS = {"Male": 0.69, "Female": 0.31}
MARRIED_PROBS = {"Yes": 0.65, "No": 0.35}
DEPENDENTS_PROBS = {"0": 0.40, "1": 0.25, "2": 0.20, "3+": 0.15}
EDUCATION_PROBS = {"Graduate": 0.78, "Not Graduate": 0.22}
SELF_EMPLOYED_PROBS = {"Yes": 0.14, "No": 0.86}
PROPERTY_AREA_PROBS = {"Urban": 0.35, "Semiurban": 0.38, "Rural": 0.27}
CREDIT_HISTORY_PROBS = {1.0: 0.85, 0.0: 0.15}

LOAN_TERM_VALUES = [12, 36, 60, 84, 120, 180, 240, 300, 360, 480]
LOAN_TERM_PROBS = [0.01, 0.02, 0.03, 0.02, 0.05, 0.10, 0.05, 0.05, 0.60, 0.07]

# Missing value rates (realistic — mimic real-world survey data)
MISSING_RATES = {
    "Gender": 0.03,
    "Married": 0.01,
    "Dependents": 0.04,
    "Self_Employed": 0.05,
    "LoanAmount": 0.04,
    "Loan_Amount_Term": 0.02,
    "Credit_History": 0.08,
}


def _sample_categorical(categories: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample from a categorical distribution."""
    labels = list(categories.keys())
    probs = list(categories.values())
    return rng.choice(labels, size=n, p=probs)


def _generate_income(education: np.ndarray, self_employed: np.ndarray,
                     n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate applicant income correlated with education and employment."""
    base = np.where(education == "Graduate", 8.4, 7.9)
    boost = np.where(self_employed == "Yes", 0.3, 0.0)
    return rng.lognormal(mean=base + boost, sigma=0.65, size=n).astype(int)


def _generate_coapplicant_income(married: np.ndarray, n: int,
                                  rng: np.random.Generator) -> np.ndarray:
    """Generate coapplicant income — higher for married applicants."""
    has_coapplicant = np.where(
        married == "Yes",
        rng.binomial(1, 0.55, size=n),
        rng.binomial(1, 0.15, size=n),
    )
    raw_income = rng.lognormal(mean=7.2, sigma=0.9, size=n)
    return (raw_income * has_coapplicant).astype(int)


def _compute_approval_probability(
    credit_history: np.ndarray,
    applicant_income: np.ndarray,
    coapplicant_income: np.ndarray,
    loan_amount: np.ndarray,
    education: np.ndarray,
    married: np.ndarray,
    property_area: np.ndarray,
    dependents: np.ndarray,
) -> np.ndarray:
    """
    Compute approval probability based on realistic financial rules.
    
    Key drivers (in order of importance):
    1. Credit history (strongest)
    2. Income-to-loan ratio
    3. Education level
    4. Marital status
    5. Property area
    """
    total_income = applicant_income + coapplicant_income
    income_to_loan = total_income / (loan_amount + 1)

    prob = np.full(len(credit_history), 0.50)

    # Credit history — strongest predictor
    prob += np.where(credit_history == 1.0, 0.25, -0.30)

    # Income-to-loan ratio
    prob += np.clip(income_to_loan * 0.005, -0.10, 0.12)

    # Education
    prob += np.where(education == "Graduate", 0.04, -0.04)

    # Married (stability signal)
    prob += np.where(married == "Yes", 0.03, -0.01)

    # Property area (semiurban has better infra + lower cost)
    prob += np.where(property_area == "Semiurban", 0.04, 0.0)
    prob += np.where(property_area == "Rural", -0.02, 0.0)

    # Dependents (more dependents = higher expenses)
    prob += np.where(dependents == "3+", -0.04, 0.0)
    prob += np.where(dependents == "0", 0.02, 0.0)

    return np.clip(prob, 0.05, 0.95)


def _inject_missing_values(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Inject realistic missing values into the dataset."""
    df = df.copy()
    for col, rate in MISSING_RATES.items():
        if col in df.columns:
            mask = rng.random(len(df)) < rate
            df.loc[mask, col] = np.nan
    return df


def generate_loan_data(n_samples: int = 8000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic loan application dataset.

    Parameters
    ----------
    n_samples : int
        Number of loan applications to generate.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Generated dataset with realistic distributions and missing values.
    """
    rng = np.random.default_rng(random_state)

    # ── Generate features ───────────────────
    loan_ids = [f"LP{str(i).zfill(6)}" for i in range(1, n_samples + 1)]

    gender = _sample_categorical(GENDER_PROBS, n_samples, rng)
    married = _sample_categorical(MARRIED_PROBS, n_samples, rng)
    dependents = _sample_categorical(DEPENDENTS_PROBS, n_samples, rng)
    education = _sample_categorical(EDUCATION_PROBS, n_samples, rng)
    self_employed = _sample_categorical(SELF_EMPLOYED_PROBS, n_samples, rng)
    property_area = _sample_categorical(PROPERTY_AREA_PROBS, n_samples, rng)
    credit_history = _sample_categorical(CREDIT_HISTORY_PROBS, n_samples, rng).astype(float)

    applicant_income = _generate_income(education, self_employed, n_samples, rng)
    coapplicant_income = _generate_coapplicant_income(married, n_samples, rng)
    loan_amount = rng.lognormal(mean=4.9, sigma=0.55, size=n_samples).astype(int)
    loan_amount_term = rng.choice(LOAN_TERM_VALUES, size=n_samples, p=LOAN_TERM_PROBS).astype(float)

    # ── Compute target ──────────────────────
    approval_prob = _compute_approval_probability(
        credit_history, applicant_income, coapplicant_income,
        loan_amount, education, married, property_area, dependents,
    )
    loan_status = rng.binomial(1, approval_prob)
    loan_status_str = np.where(loan_status == 1, "Y", "N")

    # ── Assemble DataFrame ──────────────────
    df = pd.DataFrame({
        "Loan_ID": loan_ids,
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_amount_term,
        "Credit_History": credit_history,
        "Property_Area": property_area,
        "Loan_Status": loan_status_str,
    })

    # ── Inject missing values ───────────────
    df = _inject_missing_values(df, rng)

    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic loan dataset")
    parser.add_argument("--n", type=int, default=8000, help="Number of samples (default: 8000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    output_path = Path(__file__).resolve().parent / "loan_data.csv"

    print(f"Generating {args.n} loan applications (seed={args.seed})...")
    df = generate_loan_data(n_samples=args.n, random_state=args.seed)

    df.to_csv(output_path, index=False)
    print(f"[OK] Saved to {output_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Approval rate: {(df['Loan_Status'] == 'Y').mean():.1%}")
    print(f"  Missing values: {df.isnull().sum().sum()} total")

    # Quick summary
    print(f"\n  Feature summary:")
    for col in df.columns:
        missing = df[col].isnull().sum()
        missing_str = f" ({missing} missing)" if missing > 0 else ""
        if df[col].dtype == "object":
            print(f"    {col}: {df[col].nunique()} unique values{missing_str}")
        else:
            print(f"    {col}: mean={df[col].mean():.1f}, std={df[col].std():.1f}{missing_str}")


if __name__ == "__main__":
    main()
