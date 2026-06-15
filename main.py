"""
Loan Approval Prediction — Full Pipeline
=========================================

Runs the complete ML pipeline end-to-end:
1. Load data
2. Feature engineering
3. Preprocessing (imputation, encoding, scaling)
4. Model training (with cross-validation + tuning)
5. Evaluation (metrics, plots)
6. Explainability (SHAP, feature importance)
7. Save best model

Usage:
    python main.py
"""

import logging
import sys
import time

import joblib

from src.config import (
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    BEST_MODEL_PATH,
    OUTPUT_DIR,
)

# ── Configure logging ───────────────────────
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUTPUT_DIR / "pipeline.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)


def main():
    start_time = time.time()
    logger.info("╔══════════════════════════════════════════════════╗")
    logger.info("║  LOAN APPROVAL PREDICTION — ML PIPELINE         ║")
    logger.info("╚══════════════════════════════════════════════════╝")
    logger.info("")

    # ── Step 1: Load Data ───────────────────
    from src.data_loader import load_data
    df = load_data()

    # ── Step 2: Preprocessing (includes imputation + feature engineering + encoding + scaling)
    from src.preprocessing import preprocess
    data = preprocess(df)

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    feature_names = data["feature_names"]

    # ── Step 4: Model Training ──────────────
    from src.model_training import train_all_models
    results = train_all_models(X_train, y_train)

    # ── Step 5: Evaluation ──────────────────
    from src.evaluation import evaluate_all_models, generate_all_plots
    comparison_df = evaluate_all_models(results, X_test, y_test)
    generate_all_plots(results, X_test, y_test)

    # Print comparison table
    logger.info("")
    logger.info("MODEL COMPARISON TABLE:")
    logger.info("-" * 80)
    print(comparison_df.to_string(index=False))
    logger.info("-" * 80)

    # ── Step 6: Explainability ──────────────
    # Use the best tree-based model for SHAP/feature importance
    # (LinearExplainer doesn't produce the same quality plots)
    tree_models = [r for r in results if hasattr(r["model"], "feature_importances_")]
    if tree_models:
        explain_model = tree_models[0]["model"]
        explain_name = tree_models[0]["name"]
        logger.info(f"  Using '{explain_name}' for explainability (tree-based)")
    else:
        explain_model = results[0]["model"]
        explain_name = results[0]["name"]

    from src.explainability import generate_explainability
    generate_explainability(explain_model, X_test, feature_names)

    # ── Step 7: Save Best Model ─────────────
    logger.info("=" * 50)
    logger.info("SAVING BEST MODEL")
    logger.info("=" * 50)
    joblib.dump(best_model, BEST_MODEL_PATH)
    logger.info(f"  ★ Saved best model ({results[0]['name']}) → {BEST_MODEL_PATH}")

    # ── Summary ─────────────────────────────
    elapsed = time.time() - start_time
    logger.info("")
    logger.info("╔══════════════════════════════════════════════════╗")
    logger.info("║  PIPELINE COMPLETE                               ║")
    logger.info("╚══════════════════════════════════════════════════╝")
    logger.info(f"  Total time: {elapsed:.1f}s")
    logger.info(f"  Best model: {results[0]['name']}")
    logger.info(f"  Best F1-Score: {results[0]['cv_score']:.4f}")
    logger.info(f"  Outputs saved to: {OUTPUT_DIR}")
    logger.info(f"  Model saved to: {BEST_MODEL_PATH}")
    logger.info("")
    logger.info("  Next steps:")
    logger.info("    → Run the dashboard:  streamlit run app/streamlit_app.py")
    logger.info("    → View results:       outputs/model_comparison.csv")


if __name__ == "__main__":
    main()
