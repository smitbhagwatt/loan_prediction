"""
Loan Approval Prediction — Interactive Dashboard
=================================================

Streamlit app for interactive loan approval prediction with:
- Applicant information input (sidebar)
- Real-time prediction with confidence score
- SHAP explanation for individual predictions
- Model performance overview tab
"""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import sys

# Add project root to path for imports
BASE_DIR_INIT = Path(__file__).resolve().parent.parent
if str(BASE_DIR_INIT) not in sys.path:
    sys.path.insert(0, str(BASE_DIR_INIT))

from src.database import LoanDatabase

# ── Page config ─────────────────────────────
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ───────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.joblib"
OUTPUT_DIR = BASE_DIR / "outputs"

# ── Custom CSS ──────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .main { background-color: #0e1117; }

    .stApp {
        background: linear-gradient(135deg, #0e1117 0%, #1a1a2e 50%, #16213e 100%);
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.2);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1, #22d3ee);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 0.3rem;
    }

    /* Prediction result */
    .prediction-approved {
        background: linear-gradient(135deg, #064e3b, #065f46);
        border: 2px solid #10b981;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }
    .prediction-rejected {
        background: linear-gradient(135deg, #450a0a, #7f1d1d);
        border: 2px solid #ef4444;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }
    .prediction-text {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .confidence-text {
        font-size: 1.1rem;
        color: #94a3b8;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a, #1e293b);
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Section headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #e2e8f0;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(99, 102, 241, 0.3);
    }

    /* Progress bar customization */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #6366f1, #22d3ee);
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Helper Functions ────────────────────────

@st.cache_resource
def load_model():
    """Load model and preprocessing artifacts (cached)."""
    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        return None, None
    model = joblib.load(MODEL_PATH)
    artifacts = joblib.load(PREPROCESSOR_PATH)
    return model, artifacts


def predict_loan(input_data: dict, model, artifacts) -> dict:
    """Make prediction for a single application."""
    encoders = artifacts["encoders"]
    scaler = artifacts["scaler"]
    feature_names = artifacts["feature_names"]

    df = pd.DataFrame([input_data])

    # Feature engineering
    df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["IncomeToLoan"] = df["TotalIncome"] / (df["LoanAmount"] + 1)
    df["EMI"] = df["LoanAmount"] / (df["Loan_Amount_Term"] + 1)
    df["LogLoanAmount"] = np.log1p(df["LoanAmount"])
    df["LogTotalIncome"] = np.log1p(df["TotalIncome"])

    # Encode categoricals
    cat_cols = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
    for col in cat_cols:
        if col in encoders:
            le = encoders[col]
            val = str(df[col].iloc[0])
            if val in le.classes_:
                df[col] = le.transform([val])[0]
            else:
                df[col] = 0

    # Scale numericals
    num_cols = [
        "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History",
        "TotalIncome", "IncomeToLoan", "EMI", "LogLoanAmount", "LogTotalIncome"
    ]
    cols_present = [c for c in num_cols if c in df.columns]
    df[cols_present] = scaler.transform(df[cols_present])

    # Ensure correct feature order
    df = df[feature_names]

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "prediction": "Approved" if prediction == 1 else "Rejected",
        "probability": probability,
        "features_df": df,
    }


# ── Main App ────────────────────────────────

def main():
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0 2rem 0;">
        <h1 style="font-size: 2.5rem; font-weight: 700;
                   background: linear-gradient(135deg, #6366f1, #22d3ee, #f59e0b);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            🏦 Loan Approval Predictor
        </h1>
        <p style="color: #94a3b8; font-size: 1.1rem; margin-top: -0.5rem;">
            ML-powered loan approval prediction with real-time explainability
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Load model
    model, artifacts = load_model()

    if model is None:
        st.error("⚠️ Model not found. Please run `python main.py` first to train the model.")
        st.code("python main.py", language="bash")
        return

    # ── Sidebar: Input Form ─────────────────
    with st.sidebar:
        st.markdown("### 📋 Applicant Information")
        st.markdown("---")

        st.markdown("**Personal Details**")
        gender = st.selectbox("Gender", ["Male", "Female"], index=0)
        married = st.selectbox("Marital Status", ["Yes", "No"], index=0)
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"], index=0)
        education = st.selectbox("Education", ["Graduate", "Not Graduate"], index=0)
        self_employed = st.selectbox("Self Employed", ["No", "Yes"], index=0)

        st.markdown("---")
        st.markdown("**Financial Details**")
        applicant_income = st.slider("Applicant Income (₹/mo)", 500, 50000, 5000, step=500)
        coapplicant_income = st.slider("Co-applicant Income (₹/mo)", 0, 30000, 1500, step=500)
        loan_amount = st.slider("Loan Amount (₹K)", 10, 700, 150, step=10)
        loan_term = st.selectbox("Loan Term (months)", [360, 180, 120, 84, 60, 36, 480], index=0)
        credit_history = st.selectbox("Credit History", [1.0, 0.0],
                                       format_func=lambda x: "Good (1)" if x == 1.0 else "Bad (0)")

        st.markdown("---")
        st.markdown("**Property Details**")
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"], index=1)

        predict_btn = st.button("🔍 Predict Approval", use_container_width=True, type="primary")

    # ── Tabs ────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["🔮 Prediction", "📊 Model Performance", "📁 History & Reports", "ℹ️ About"])

    with tab1:
        if predict_btn:
            input_data = {
                "Gender": gender,
                "Married": married,
                "Dependents": dependents,
                "Education": education,
                "Self_Employed": self_employed,
                "ApplicantIncome": applicant_income,
                "CoapplicantIncome": coapplicant_income,
                "LoanAmount": loan_amount,
                "Loan_Amount_Term": loan_term,
                "Credit_History": credit_history,
                "Property_Area": property_area,
            }

            with st.spinner("Analyzing application..."):
                result = predict_loan(input_data, model, artifacts)

            # Save prediction to SQLite database
            try:
                db = LoanDatabase()
                app_id = db.save_prediction(
                    input_data, result["prediction"], result["probability"]
                )
                st.toast(f"✅ Application #{app_id} saved to database", icon="💾")
            except Exception as e:
                st.toast(f"⚠️ Could not save to database: {e}", icon="⚠️")

            # Prediction Result
            is_approved = result["prediction"] == "Approved"
            prob = result["probability"]

            css_class = "prediction-approved" if is_approved else "prediction-rejected"
            emoji = "✅" if is_approved else "❌"
            color = "#10b981" if is_approved else "#ef4444"

            st.markdown(f"""
            <div class="{css_class}">
                <div class="prediction-text" style="color: {color};">
                    {emoji} Loan {result['prediction']}
                </div>
                <div class="confidence-text">
                    Approval Probability: {prob:.1%}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                total_income = applicant_income + coapplicant_income
                st.metric("Total Income", f"₹{total_income:,.0f}/mo")
            with col2:
                emi = loan_amount / (loan_term + 1) * 1000
                st.metric("Est. EMI", f"₹{emi:,.0f}/mo")
            with col3:
                ratio = total_income / (loan_amount + 1)
                st.metric("Income/Loan Ratio", f"{ratio:.1f}x")
            with col4:
                risk = "Low" if prob > 0.7 else ("Medium" if prob > 0.4 else "High")
                st.metric("Risk Level", risk)

            # SHAP explanation
            st.markdown('<div class="section-header">🧠 Why this decision?</div>',
                        unsafe_allow_html=True)

            try:
                import shap
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(result["features_df"])

                if isinstance(shap_values, list):
                    shap_values = shap_values[1]

                fig, ax = plt.subplots(figsize=(10, 4))
                fig.patch.set_facecolor('#0e1117')
                ax.set_facecolor('#0e1117')

                shap.waterfall_plot(
                    shap.Explanation(
                        values=shap_values[0],
                        base_values=explainer.expected_value if not isinstance(explainer.expected_value, list) else explainer.expected_value[1],
                        data=result["features_df"].iloc[0].values,
                        feature_names=artifacts["feature_names"],
                    ),
                    show=False,
                    max_display=10,
                )
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            except Exception as e:
                st.info(f"SHAP explanation not available for this model type. ({e})")

            # Application Summary
            st.markdown('<div class="section-header">📋 Application Summary</div>',
                        unsafe_allow_html=True)

            summary_data = {
                "Field": ["Gender", "Marital Status", "Dependents", "Education",
                          "Self Employed", "Applicant Income", "Co-applicant Income",
                          "Loan Amount", "Loan Term", "Credit History", "Property Area"],
                "Value": [gender, married, dependents, education, self_employed,
                          f"₹{applicant_income:,}", f"₹{coapplicant_income:,}",
                          f"₹{loan_amount}K", f"{loan_term} months",
                          "Good" if credit_history == 1.0 else "Bad", property_area],
            }
            st.table(pd.DataFrame(summary_data))

        else:
            st.markdown("""
            <div style="text-align: center; padding: 4rem 2rem; color: #64748b;">
                <p style="font-size: 3rem;">👈</p>
                <h3>Fill in applicant details</h3>
                <p>Enter loan application information in the sidebar and click <strong>Predict Approval</strong></p>
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="section-header">📊 Model Performance Comparison</div>',
                    unsafe_allow_html=True)

        comparison_path = OUTPUT_DIR / "model_comparison.csv"
        if comparison_path.exists():
            comp_df = pd.read_csv(comparison_path)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
        else:
            st.info("Run `python main.py` to generate model comparison data.")

        # Show plots
        col1, col2 = st.columns(2)

        roc_path = OUTPUT_DIR / "roc_curves.png"
        cm_path = OUTPUT_DIR / "confusion_matrix.png"
        fi_path = OUTPUT_DIR / "feature_importance.png"
        shap_path = OUTPUT_DIR / "shap_summary.png"

        if roc_path.exists():
            with col1:
                st.markdown('<div class="section-header">ROC Curves</div>',
                            unsafe_allow_html=True)
                st.image(str(roc_path))

        if cm_path.exists():
            with col2:
                st.markdown('<div class="section-header">Confusion Matrices</div>',
                            unsafe_allow_html=True)
                st.image(str(cm_path))

        if fi_path.exists():
            st.markdown('<div class="section-header">Feature Importance</div>',
                        unsafe_allow_html=True)
            st.image(str(fi_path))

        if shap_path.exists():
            st.markdown('<div class="section-header">SHAP Analysis</div>',
                        unsafe_allow_html=True)
            st.image(str(shap_path))

    with tab3:
        st.markdown('<div class="section-header">📁 Prediction History & SQL Reports</div>',
                    unsafe_allow_html=True)

        try:
            db = LoanDatabase()
            total = db.get_total_count()

            if total == 0:
                st.info("No predictions yet. Use the **Prediction** tab to make your first prediction — it will be saved automatically.")
            else:
                # ── Summary Metrics ──────────────────
                stats = db.get_approval_stats()
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Total Applications", stats["total_applications"])
                with m2:
                    st.metric("Approved", stats["approved_count"])
                with m3:
                    st.metric("Rejected", stats["rejected_count"])
                with m4:
                    st.metric("Avg Approval Prob", f"{stats['avg_approval_prob']}%")

                st.markdown("<br>", unsafe_allow_html=True)

                # ── Charts Row ───────────────────────
                chart_col1, chart_col2 = st.columns(2)

                with chart_col1:
                    st.markdown('<div class="section-header">Approval by Property Area</div>',
                                unsafe_allow_html=True)
                    area_stats = db.get_stats_by_property_area()
                    if area_stats:
                        area_df = pd.DataFrame(area_stats)
                        st.dataframe(area_df.rename(columns={
                            "property_area": "Area",
                            "total": "Total",
                            "approved": "Approved",
                            "approval_rate": "Approval Rate (%)",
                            "avg_loan": "Avg Loan (₹K)"
                        }), use_container_width=True, hide_index=True)

                with chart_col2:
                    st.markdown('<div class="section-header">Approval by Education</div>',
                                unsafe_allow_html=True)
                    edu_stats = db.get_stats_by_education()
                    if edu_stats:
                        edu_df = pd.DataFrame(edu_stats)
                        st.dataframe(edu_df.rename(columns={
                            "education": "Education",
                            "total": "Total",
                            "approved": "Approved",
                            "approval_rate": "Approval Rate (%)"
                        }), use_container_width=True, hide_index=True)

                # ── Risk Distribution ────────────────
                st.markdown('<div class="section-header">Risk Distribution</div>',
                            unsafe_allow_html=True)
                risk_stats = db.get_risk_distribution()
                if risk_stats:
                    risk_df = pd.DataFrame(risk_stats)
                    st.dataframe(risk_df.rename(columns={
                        "risk_level": "Risk Level",
                        "count": "Count",
                        "avg_probability": "Avg Probability (%)"
                    }), use_container_width=True, hide_index=True)

                # ── Full History Table ───────────────
                st.markdown('<div class="section-header">Recent Predictions</div>',
                            unsafe_allow_html=True)
                history = db.get_prediction_history(limit=50)
                if history:
                    hist_df = pd.DataFrame(history)
                    display_cols = {
                        "application_id": "#",
                        "gender": "Gender",
                        "education": "Education",
                        "property_area": "Area",
                        "applicant_income": "Income (₹)",
                        "loan_amount": "Loan (₹K)",
                        "credit_history": "Credit",
                        "prediction": "Result",
                        "approval_probability": "Probability",
                        "risk_level": "Risk",
                        "created_at": "Date",
                    }
                    cols_available = [c for c in display_cols if c in hist_df.columns]
                    st.dataframe(
                        hist_df[cols_available].rename(columns=display_cols),
                        use_container_width=True,
                        hide_index=True,
                    )

        except Exception as e:
            st.error(f"Database error: {e}")

    with tab4:
        st.markdown("""
        ### About This Project

        **Loan Approval Prediction** is an end-to-end machine learning project that predicts
        whether a loan application should be approved or rejected based on applicant
        demographic and financial information.

        #### Pipeline Overview
        1. **Data Loading** — Validate and profile the dataset
        2. **Feature Engineering** — Create derived financial features (TotalIncome, EMI, IncomeToLoan)
        3. **Preprocessing** — Impute missing values, encode categoricals, scale numericals
        4. **Model Training** — Train 4 models with cross-validation and hyperparameter tuning
        5. **Evaluation** — Compare models on accuracy, precision, recall, F1, ROC-AUC
        6. **Explainability** — SHAP analysis for model transparency
        7. **Database** — SQLite storage for historical analysis and SQL-based reporting

        #### Models Trained
        - Logistic Regression (baseline)
        - Decision Tree Classifier
        - Random Forest Classifier (tuned)
        - XGBoost Classifier (tuned)

        #### Tech Stack
        `Python` · `Scikit-Learn` · `XGBoost` · `SHAP` · `Pandas` · `Streamlit` · `SQLite`

        ---
        *Built as an internship project demonstrating end-to-end ML engineering.*
        """)


if __name__ == "__main__":
    main()
