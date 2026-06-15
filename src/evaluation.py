"""
Model Evaluation
================

Evaluates trained models on the test set, generates comparison tables,
and produces publication-quality diagnostic plots.

Outputs:
- model_comparison.csv — side-by-side metrics for all models
- confusion_matrix.png — confusion matrices for all models
- roc_curves.png — overlaid ROC curves
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

from src.config import OUTPUT_DIR

logger = logging.getLogger(__name__)

# ── Plot styling ────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0e1117",
    "axes.facecolor": "#0e1117",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#fafafa",
    "text.color": "#fafafa",
    "xtick.color": "#cccccc",
    "ytick.color": "#cccccc",
    "grid.color": "#1e2530",
    "figure.dpi": 150,
    "font.family": "sans-serif",
})

PALETTE = ["#6366f1", "#22d3ee", "#f59e0b", "#ef4444"]


def evaluate_single_model(name: str, model, X_test, y_test) -> dict:
    """Evaluate a single model and return all metrics."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_proba) if y_proba is not None else None,
    }

    logger.info(f"  {name}:")
    logger.info(f"    Accuracy={metrics['Accuracy']:.4f}  Precision={metrics['Precision']:.4f}  "
                f"Recall={metrics['Recall']:.4f}  F1={metrics['F1-Score']:.4f}  "
                f"AUC={metrics['ROC-AUC']:.4f}")

    return metrics


def evaluate_all_models(results: list[dict], X_test, y_test) -> pd.DataFrame:
    """
    Evaluate all models and produce a comparison DataFrame.

    Parameters
    ----------
    results : list[dict]
        Output from train_all_models(). Each dict has 'name' and 'model'.
    X_test, y_test : array-like
        Test data.

    Returns
    -------
    pd.DataFrame
        Comparison table with one row per model.
    """
    logger.info("=" * 50)
    logger.info("MODEL EVALUATION (Test Set)")
    logger.info("=" * 50)

    all_metrics = []
    for r in results:
        metrics = evaluate_single_model(r["name"], r["model"], X_test, y_test)
        metrics["CV F1-Score"] = r["cv_score"]
        metrics["Train Time (s)"] = round(r["train_time"], 2)
        all_metrics.append(metrics)

    comparison_df = pd.DataFrame(all_metrics)
    comparison_df = comparison_df.sort_values("F1-Score", ascending=False).reset_index(drop=True)

    # Save to CSV
    output_path = OUTPUT_DIR / "model_comparison.csv"
    comparison_df.to_csv(output_path, index=False)
    logger.info(f"\n  Saved comparison table → {output_path}")

    return comparison_df


def plot_confusion_matrices(results: list[dict], X_test, y_test) -> None:
    """Plot confusion matrices for all models in a grid."""
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5))

    if n_models == 1:
        axes = [axes]

    for ax, r, color in zip(axes, results, PALETTE):
        y_pred = r["model"].predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=["Rejected", "Approved"],
            yticklabels=["Rejected", "Approved"],
            cbar=False,
            annot_kws={"size": 14, "weight": "bold"},
        )
        ax.set_title(r["name"], fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual", fontsize=10)

    fig.suptitle("Confusion Matrices", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    output_path = OUTPUT_DIR / "confusion_matrix.png"
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info(f"  Saved confusion matrices → {output_path}")


def plot_roc_curves(results: list[dict], X_test, y_test) -> None:
    """Plot overlaid ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for r, color in zip(results, PALETTE):
        if hasattr(r["model"], "predict_proba"):
            y_proba = r["model"].predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc_val = roc_auc_score(y_test, y_proba)
            ax.plot(fpr, tpr, color=color, lw=2.5,
                    label=f"{r['name']} (AUC={auc_val:.3f})")

    # Diagonal reference line
    ax.plot([0, 1], [0, 1], color="#555555", lw=1.5, linestyle="--", label="Random (AUC=0.500)")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Model Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10, facecolor="#1a1a2e", edgecolor="#333")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.2)

    output_path = OUTPUT_DIR / "roc_curves.png"
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info(f"  Saved ROC curves → {output_path}")


def generate_all_plots(results: list[dict], X_test, y_test) -> None:
    """Generate all evaluation plots."""
    logger.info("Generating evaluation plots...")
    plot_confusion_matrices(results, X_test, y_test)
    plot_roc_curves(results, X_test, y_test)
