"""
Prediction Module
=================

Load saved model and preprocessing artifacts to make predictions
on new loan applications.
"""

import logging

import joblib
import numpy as np
import pandas as pd

from src.config import BEST_MODEL_PATH, PREPROCESSOR_PATH

logger = logging.getLogger(__name__)


def load_model():
    """Load the saved best model and preprocessor."""
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {BEST_MODEL_PATH}. "
            f"Run 'python main.py' to train models first."
        )

    model = joblib.load(BEST_MODEL_PATH)
    artifacts = joblib.load(PREPROCESSOR_PATH)

    logger.info(f"Loaded model from {BEST_MODEL_PATH}")
    return model, artifacts


def predict_single(input_data: dict) -> dict:
    """
    Make a prediction for a single loan application.

    Parameters
    ----------
    input_data : dict
        Raw loan application data. Example:
        {
            "Gender": "Male",
            "Married": "Yes",
            "Dependents": "1",
            "Education": "Graduate",
            "Self_Employed": "No",
            "ApplicantIncome": 5000,
            "CoapplicantIncome": 2000,
            "LoanAmount": 150,
            "Loan_Amount_Term": 360,
            "Credit_History": 1.0,
            "Property_Area": "Urban",
        }

    Returns
    -------
    dict with keys:
        prediction: str ("Approved" or "Rejected")
        probability: float (approval probability 0-1)
        confidence: str ("High", "Medium", "Low")
    """
    model, artifacts = load_model()
    encoders = artifacts["encoders"]
    scaler = artifacts["scaler"]
    feature_names = artifacts["feature_names"]

    # Create DataFrame
    df = pd.DataFrame([input_data])

    # Feature engineering
    df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["IncomeToLoan"] = df["TotalIncome"] / (df["LoanAmount"] + 1)
    df["EMI"] = df["LoanAmount"] / (df["Loan_Amount_Term"] + 1)
    df["LogLoanAmount"] = np.log1p(df["LoanAmount"])
    df["LogTotalIncome"] = np.log1p(df["TotalIncome"])

    # Encode categoricals
    from src.config import CATEGORICAL_COLS, NUMERICAL_COLS
    for col in CATEGORICAL_COLS:
        if col in encoders:
            le = encoders[col]
            val = str(df[col].iloc[0])
            if val in le.classes_:
                df[col] = le.transform([val])[0]
            else:
                df[col] = 0  # Unknown category → default

    # Scale numericals
    num_cols = [c for c in NUMERICAL_COLS if c in df.columns]
    df[num_cols] = scaler.transform(df[num_cols])

    # Ensure correct column order
    df = df[feature_names]

    # Predict
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1] if hasattr(model, "predict_proba") else float(prediction)

    # Confidence level
    if probability > 0.8 or probability < 0.2:
        confidence = "High"
    elif probability > 0.6 or probability < 0.4:
        confidence = "Medium"
    else:
        confidence = "Low"

    result = {
        "prediction": "Approved" if prediction == 1 else "Rejected",
        "probability": round(probability, 4),
        "confidence": confidence,
    }

    logger.info(f"Prediction: {result['prediction']} (prob={result['probability']:.1%}, "
                f"confidence={result['confidence']})")

    return result
