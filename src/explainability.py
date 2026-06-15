"""
Model Explainability
====================

Generates interpretable explanations for model predictions using
SHAP (SHapley Additive exPlanations) and feature importance analysis.

Outputs:
- feature_importance.png — bar chart of feature importances
- shap_summary.png — SHAP beeswarm plot
"""

import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.config import OUTPUT_DIR

logger = logging.getLogger(__name__)

# Suppress SHAP's verbose output
warnings.filterwarnings("ignore", category=FutureWarning)


def plot_feature_importance(model, feature_names: list, top_n: int = 15) -> None:
    """
    Plot feature importance from a tree-based model.

    Parameters
    ----------
    model : estimator
        A fitted model with `feature_importances_` attribute.
    feature_names : list
        Feature names matching the model's input.
    top_n : int
        Number of top features to display.
    """
    if not hasattr(model, "feature_importances_"):
        logger.warning("  Model does not have feature_importances_. Skipping.")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(indices)))

    ax.barh(
        range(len(indices)),
        importances[indices],
        color=colors,
        edgecolor="#333",
        height=0.7,
    )
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=11)
    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_title("Feature Importance (Best Model)", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.2)

    # Add value labels
    for i, (val, idx) in enumerate(zip(importances[indices], indices)):
        ax.text(val + 0.003, i, f"{val:.3f}", va="center", fontsize=9, color="#cccccc")

    plt.tight_layout()
    output_path = OUTPUT_DIR / "feature_importance.png"
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info(f"  Saved feature importance → {output_path}")


def plot_shap_summary(model, X_test, feature_names: list) -> None:
    """
    Generate SHAP summary plot (beeswarm) for the best model.

    Parameters
    ----------
    model : estimator
        Fitted model compatible with SHAP TreeExplainer.
    X_test : array-like
        Test features (use a subsample for speed).
    feature_names : list
        Feature names.
    """
    logger.info("  Computing SHAP values (this may take a moment)...")

    try:
        # Use TreeExplainer for tree-based models
        explainer = shap.TreeExplainer(model)

        # Subsample for speed if dataset is large
        if len(X_test) > 500:
            X_sample = X_test.sample(500, random_state=42) if isinstance(X_test, pd.DataFrame) else X_test[:500]
        else:
            X_sample = X_test

        shap_values = explainer.shap_values(X_sample)

        # Handle binary classification (shap_values might be a list)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Take positive class

        # Create summary plot
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(
            shap_values,
            X_sample,
            feature_names=feature_names,
            show=False,
            plot_size=None,
            max_display=15,
        )
        plt.title("SHAP Feature Impact on Loan Approval", fontsize=14, fontweight="bold")
        plt.tight_layout()

        output_path = OUTPUT_DIR / "shap_summary.png"
        plt.savefig(output_path, bbox_inches="tight", dpi=150)
        plt.close("all")
        logger.info(f"  Saved SHAP summary → {output_path}")

    except Exception as e:
        logger.warning(f"  SHAP analysis failed: {e}")
        logger.warning("  This is non-critical — other outputs are still generated.")


def generate_explainability(model, X_test, feature_names: list) -> None:
    """Run all explainability analyses."""
    logger.info("=" * 50)
    logger.info("EXPLAINABILITY ANALYSIS")
    logger.info("=" * 50)

    plot_feature_importance(model, feature_names)
    plot_shap_summary(model, X_test, feature_names)
