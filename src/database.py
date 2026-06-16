"""
Database Module — SQLite Backend
=================================

Provides a SQLite database layer for the Loan Approval Prediction project.
Stores loan applications, prediction outcomes, and applicant profiles,
enabling SQL-based reporting and historical analysis.

Tables:
    - applicants: Stores applicant demographic profiles
    - loan_applications: Stores loan application details and prediction outcomes
    - model_runs: Stores model training run metadata

Usage:
    from src.database import LoanDatabase
    db = LoanDatabase()
    db.save_prediction(input_data, prediction, probability)
    history = db.get_prediction_history()
"""

import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager

from src.config import BASE_DIR

logger = logging.getLogger(__name__)

# Database file path
DB_PATH = BASE_DIR / "data" / "loan_predictions.db"


class LoanDatabase:
    """SQLite database manager for loan prediction storage and reporting."""

    def __init__(self, db_path: str = None):
        """Initialize database connection and create tables if needed."""
        self.db_path = db_path or str(DB_PATH)
        self._create_tables()

    @contextmanager
    def _get_connection(self):
        """Context manager for safe database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _create_tables(self):
        """Create database schema if tables don't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # ── Table 1: Applicant Profiles ──────────────
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS applicants (
                    applicant_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                    gender          TEXT NOT NULL,
                    married         TEXT NOT NULL,
                    dependents      TEXT NOT NULL,
                    education       TEXT NOT NULL,
                    self_employed   TEXT NOT NULL,
                    property_area   TEXT NOT NULL,
                    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # ── Table 2: Loan Applications & Predictions ─
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS loan_applications (
                    application_id      INTEGER PRIMARY KEY AUTOINCREMENT,
                    applicant_id        INTEGER NOT NULL,
                    applicant_income    REAL NOT NULL,
                    coapplicant_income  REAL NOT NULL,
                    loan_amount         REAL NOT NULL,
                    loan_amount_term    REAL NOT NULL,
                    credit_history      REAL NOT NULL,
                    prediction          TEXT NOT NULL,
                    approval_probability REAL NOT NULL,
                    risk_level          TEXT NOT NULL,
                    total_income        REAL,
                    emi                 REAL,
                    income_to_loan      REAL,
                    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (applicant_id) REFERENCES applicants(applicant_id)
                )
            """)

            # ── Table 3: Model Training Runs ─────────────
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_runs (
                    run_id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name      TEXT NOT NULL,
                    accuracy        REAL,
                    precision_score REAL,
                    recall          REAL,
                    f1_score        REAL,
                    roc_auc         REAL,
                    is_best_model   INTEGER DEFAULT 0,
                    training_date   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # ── Indexes for faster queries ───────────────
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_applications_date
                ON loan_applications(created_at)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_applications_prediction
                ON loan_applications(prediction)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_applicants_area
                ON applicants(property_area)
            """)

        logger.info(f"  Database initialized at {self.db_path}")

    # ────────────────────────────────────────────
    # INSERT Operations
    # ────────────────────────────────────────────

    def save_prediction(self, input_data: dict, prediction: str,
                        probability: float) -> int:
        """
        Save a loan application and its prediction to the database.

        Parameters
        ----------
        input_data : dict
            Raw applicant input data from the Streamlit form.
        prediction : str
            'Approved' or 'Rejected'.
        probability : float
            Model's approval probability (0-1).

        Returns
        -------
        int
            The application_id of the saved record.
        """
        # Compute derived fields
        total_income = (input_data["ApplicantIncome"]
                        + input_data["CoapplicantIncome"])
        emi = input_data["LoanAmount"] / (input_data["Loan_Amount_Term"] + 1)
        income_to_loan = total_income / (input_data["LoanAmount"] + 1)
        risk_level = ("Low" if probability > 0.7
                      else ("Medium" if probability > 0.4 else "High"))

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Insert applicant profile
            cursor.execute("""
                INSERT INTO applicants
                    (gender, married, dependents, education,
                     self_employed, property_area)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                input_data["Gender"],
                input_data["Married"],
                input_data["Dependents"],
                input_data["Education"],
                input_data["Self_Employed"],
                input_data["Property_Area"],
            ))
            applicant_id = cursor.lastrowid

            # Insert loan application with prediction
            cursor.execute("""
                INSERT INTO loan_applications
                    (applicant_id, applicant_income, coapplicant_income,
                     loan_amount, loan_amount_term, credit_history,
                     prediction, approval_probability, risk_level,
                     total_income, emi, income_to_loan)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                applicant_id,
                input_data["ApplicantIncome"],
                input_data["CoapplicantIncome"],
                input_data["LoanAmount"],
                input_data["Loan_Amount_Term"],
                input_data["Credit_History"],
                prediction,
                probability,
                risk_level,
                total_income,
                emi,
                income_to_loan,
            ))
            application_id = cursor.lastrowid

        logger.info(
            f"  Saved prediction #{application_id}: {prediction} "
            f"(prob={probability:.2%})"
        )
        return application_id

    def save_model_run(self, model_name: str, metrics: dict,
                       is_best: bool = False) -> int:
        """Save a model training run with its evaluation metrics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO model_runs
                    (model_name, accuracy, precision_score, recall,
                     f1_score, roc_auc, is_best_model)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                model_name,
                metrics.get("accuracy"),
                metrics.get("precision"),
                metrics.get("recall"),
                metrics.get("f1_score"),
                metrics.get("roc_auc"),
                1 if is_best else 0,
            ))
            return cursor.lastrowid

    # ────────────────────────────────────────────
    # SELECT Operations (Reporting Queries)
    # ────────────────────────────────────────────

    def get_prediction_history(self, limit: int = 50) -> list[dict]:
        """
        Retrieve recent prediction history with applicant details.

        SQL Join: loan_applications ⟕ applicants
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    la.application_id,
                    a.gender,
                    a.married,
                    a.education,
                    a.property_area,
                    la.applicant_income,
                    la.coapplicant_income,
                    la.loan_amount,
                    la.loan_amount_term,
                    la.credit_history,
                    la.prediction,
                    la.approval_probability,
                    la.risk_level,
                    la.total_income,
                    la.emi,
                    la.created_at
                FROM loan_applications la
                JOIN applicants a ON la.applicant_id = a.applicant_id
                ORDER BY la.created_at DESC
                LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_approval_stats(self) -> dict:
        """
        Aggregate approval/rejection statistics.

        Returns summary counts and rates.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    COUNT(*) as total_applications,
                    SUM(CASE WHEN prediction = 'Approved' THEN 1 ELSE 0 END)
                        as approved_count,
                    SUM(CASE WHEN prediction = 'Rejected' THEN 1 ELSE 0 END)
                        as rejected_count,
                    ROUND(AVG(approval_probability) * 100, 1)
                        as avg_approval_prob,
                    ROUND(AVG(loan_amount), 1)
                        as avg_loan_amount,
                    ROUND(AVG(total_income), 1)
                        as avg_total_income
                FROM loan_applications
            """)
            row = cursor.fetchone()
            if row and row["total_applications"] > 0:
                return dict(row)
            return {
                "total_applications": 0,
                "approved_count": 0,
                "rejected_count": 0,
                "avg_approval_prob": 0,
                "avg_loan_amount": 0,
                "avg_total_income": 0,
            }

    def get_stats_by_property_area(self) -> list[dict]:
        """Approval rates grouped by property area."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    a.property_area,
                    COUNT(*) as total,
                    SUM(CASE WHEN la.prediction = 'Approved' THEN 1 ELSE 0 END)
                        as approved,
                    ROUND(
                        AVG(CASE WHEN la.prediction = 'Approved'
                            THEN 1.0 ELSE 0.0 END) * 100, 1
                    ) as approval_rate,
                    ROUND(AVG(la.loan_amount), 1) as avg_loan
                FROM loan_applications la
                JOIN applicants a ON la.applicant_id = a.applicant_id
                GROUP BY a.property_area
                ORDER BY approval_rate DESC
            """)
            return [dict(row) for row in cursor.fetchall()]

    def get_stats_by_education(self) -> list[dict]:
        """Approval rates grouped by education level."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    a.education,
                    COUNT(*) as total,
                    SUM(CASE WHEN la.prediction = 'Approved' THEN 1 ELSE 0 END)
                        as approved,
                    ROUND(
                        AVG(CASE WHEN la.prediction = 'Approved'
                            THEN 1.0 ELSE 0.0 END) * 100, 1
                    ) as approval_rate
                FROM loan_applications la
                JOIN applicants a ON la.applicant_id = a.applicant_id
                GROUP BY a.education
            """)
            return [dict(row) for row in cursor.fetchall()]

    def get_risk_distribution(self) -> list[dict]:
        """Count of applications by risk level."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    risk_level,
                    COUNT(*) as count,
                    ROUND(AVG(approval_probability) * 100, 1)
                        as avg_probability
                FROM loan_applications
                GROUP BY risk_level
                ORDER BY
                    CASE risk_level
                        WHEN 'Low' THEN 1
                        WHEN 'Medium' THEN 2
                        WHEN 'High' THEN 3
                    END
            """)
            return [dict(row) for row in cursor.fetchall()]

    def get_model_runs(self) -> list[dict]:
        """Retrieve all model training run records."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM model_runs
                ORDER BY training_date DESC
            """)
            return [dict(row) for row in cursor.fetchall()]

    def get_total_count(self) -> int:
        """Get total number of predictions stored."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as cnt FROM loan_applications")
            row = cursor.fetchone()
            return row["cnt"] if row else 0
