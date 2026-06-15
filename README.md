# 🏦 Loan Approval Prediction

An end-to-end machine learning pipeline that predicts loan approval decisions using applicant demographic and financial data. Features cross-validated model training, hyperparameter tuning, SHAP explainability, and an interactive Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5-f7931e?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-006600?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-1.37-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-0.45-purple?style=for-the-badge)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Results](#results)
- [Setup & Usage](#setup--usage)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Future Improvements](#future-improvements)

---

## Overview

This project implements a complete ML workflow for binary classification — predicting whether a loan application should be **Approved** or **Rejected** based on applicant information.

### Key Features
- **4 models compared**: Logistic Regression, Decision Tree, Random Forest, XGBoost
- **Stratified K-Fold cross-validation** with hyperparameter tuning (GridSearchCV)
- **Feature engineering**: TotalIncome, IncomeToLoan ratio, EMI, log transforms
- **SHAP explainability** for model transparency
- **Interactive Streamlit dashboard** with real-time predictions

---

## Architecture

```
Data Collection → Feature Engineering → Preprocessing → Model Training → Evaluation → Deployment
                        ↓                      ↓              ↓              ↓
                  5 new features        Imputation       GridSearchCV    SHAP + ROC
                  (ratios, logs)        Encoding         5-Fold CV       Confusion Matrix
                                        Scaling          4 models
```

---

## Dataset

- **Size**: 8,000 loan applications (synthetic, realistic distributions)
- **Features**: 11 input features (6 categorical + 5 numerical)
- **Target**: Loan_Status (Approved/Rejected)

| Feature | Type | Description |
|---------|------|-------------|
| Gender | Categorical | Male / Female |
| Married | Categorical | Yes / No |
| Dependents | Categorical | 0 / 1 / 2 / 3+ |
| Education | Categorical | Graduate / Not Graduate |
| Self_Employed | Categorical | Yes / No |
| ApplicantIncome | Numerical | Monthly income (₹) |
| CoapplicantIncome | Numerical | Co-applicant monthly income (₹) |
| LoanAmount | Numerical | Loan amount (₹K) |
| Loan_Amount_Term | Numerical | Term in months |
| Credit_History | Numerical | Credit history (1=Good, 0=Bad) |
| Property_Area | Categorical | Urban / Semiurban / Rural |

### Engineered Features
| Feature | Formula | Signal |
|---------|---------|--------|
| TotalIncome | ApplicantIncome + CoapplicantIncome | Household capacity |
| IncomeToLoan | TotalIncome / LoanAmount | Repayment ability |
| EMI | LoanAmount / Loan_Amount_Term | Monthly burden |
| LogLoanAmount | log(1 + LoanAmount) | Reduce skewness |
| LogTotalIncome | log(1 + TotalIncome) | Reduce skewness |

---

## Results

*Results are generated after running the pipeline. Run `python main.py` to populate.*

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| XGBoost | — | — | — | — | — |
| Random Forest | — | — | — | — | — |
| Decision Tree | — | — | — | — | — |
| Logistic Regression | — | — | — | — | — |

> After running the pipeline, check `outputs/model_comparison.csv` for actual results.

---

## Setup & Usage

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/loan-approval-prediction.git
cd loan-approval-prediction

# Install dependencies
pip install -r requirements.txt
```

### Generate Dataset

```bash
python data/generate_data.py
# Generates 8,000 realistic loan applications → data/loan_data.csv
```

### Run the ML Pipeline

```bash
python main.py
# Trains all models, generates plots, saves best model
# Outputs saved to: outputs/ and models/
```

### Launch the Dashboard

```bash
streamlit run app/streamlit_app.py
# Opens interactive prediction dashboard at http://localhost:8501
```

### Run EDA (Interactive)

Open `notebooks/EDA.py` in VS Code and run cells interactively, or:

```bash
python notebooks/EDA.py
```

---

## Project Structure

```
Loan Prediction/
├── data/
│   ├── generate_data.py        # Synthetic data generator
│   └── loan_data.csv           # Generated dataset
│
├── src/
│   ├── config.py               # Central configuration
│   ├── data_loader.py          # Data loading + validation
│   ├── preprocessing.py        # Missing values, encoding, scaling
│   ├── feature_engineering.py  # Derived features
│   ├── model_training.py       # Training with CV + tuning
│   ├── evaluation.py           # Metrics + diagnostic plots
│   ├── explainability.py       # SHAP analysis
│   └── predict.py              # Inference module
│
├── models/
│   ├── best_model.joblib       # Saved best model
│   └── preprocessor.joblib     # Saved preprocessing artifacts
│
├── outputs/
│   ├── model_comparison.csv    # Metrics comparison table
│   ├── confusion_matrix.png    # Confusion matrices
│   ├── roc_curves.png          # ROC curves
│   ├── feature_importance.png  # Feature importance
│   └── shap_summary.png        # SHAP beeswarm plot
│
├── app/
│   └── streamlit_app.py        # Interactive dashboard
│
├── notebooks/
│   └── EDA.py                  # Exploratory data analysis
│
├── .streamlit/
│   └── config.toml             # Streamlit theme config
│
├── main.py                     # Pipeline orchestrator
├── requirements.txt            # Dependencies
├── Procfile                    # Deployment config
└── README.md                   # This file
```

---

## Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.10+ |
| **Data** | Pandas, NumPy |
| **ML** | Scikit-Learn, XGBoost |
| **Visualization** | Matplotlib, Seaborn |
| **Explainability** | SHAP |
| **Deployment** | Streamlit |
| **Serialization** | Joblib |

---

## Future Improvements

- [ ] Add real-world dataset from Kaggle/Lending Club
- [ ] Implement SMOTE for handling class imbalance
- [ ] Add LightGBM and CatBoost models
- [ ] Neural network baseline (simple MLP)
- [ ] API endpoint with FastAPI
- [ ] Docker containerization
- [ ] CI/CD with GitHub Actions
- [ ] A/B testing framework for model comparison

---

## License

This project is for educational and portfolio purposes.

---

*Built as part of a data science internship project.*
