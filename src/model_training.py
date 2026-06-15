"""
Model Training
==============

Trains multiple classifiers with Stratified K-Fold Cross-Validation
and optional hyperparameter tuning via GridSearchCV.

Models:
1. Logistic Regression (baseline)
2. Decision Tree
3. Random Forest (tuned)
4. XGBoost (tuned)
"""

import logging
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from src.config import (
    RANDOM_SEED,
    CV_FOLDS,
    LOGISTIC_REGRESSION_PARAMS,
    DECISION_TREE_PARAMS,
    RANDOM_FOREST_PARAMS,
    XGBOOST_PARAMS,
)

logger = logging.getLogger(__name__)


def _get_models() -> dict:
    """Return model instances with their hyperparameter grids."""
    return {
        "Logistic Regression": {
            "model": LogisticRegression(random_state=RANDOM_SEED),
            "params": LOGISTIC_REGRESSION_PARAMS,
            "tune": True,
        },
        "Decision Tree": {
            "model": DecisionTreeClassifier(random_state=RANDOM_SEED),
            "params": DECISION_TREE_PARAMS,
            "tune": True,
        },
        "Random Forest": {
            "model": RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1),
            "params": RANDOM_FOREST_PARAMS,
            "tune": True,
        },
        "XGBoost": {
            "model": XGBClassifier(
                random_state=RANDOM_SEED,
                eval_metric="logloss",
                use_label_encoder=False,
                verbosity=0,
            ),
            "params": XGBOOST_PARAMS,
            "tune": True,
        },
    }


def train_single_model(
    name: str,
    model,
    params: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    tune: bool = True,
) -> dict:
    """
    Train a single model with optional GridSearchCV.

    Parameters
    ----------
    name : str
        Model name for logging.
    model : estimator
        Scikit-learn compatible model.
    params : dict
        Hyperparameter grid for GridSearchCV.
    X_train, y_train : array-like
        Training data.
    tune : bool
        Whether to perform hyperparameter tuning.

    Returns
    -------
    dict with keys: name, model, best_params, cv_score, cv_std, train_time
    """
    logger.info(f"  Training: {name}")
    start = time.time()

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    if tune and params:
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=params,
            cv=cv,
            scoring="f1",
            n_jobs=-1,
            verbose=0,
            refit=True,
        )
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        cv_score = grid_search.best_score_
        cv_std = grid_search.cv_results_["std_test_score"][grid_search.best_index_]
    else:
        from sklearn.model_selection import cross_val_score

        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
        model.fit(X_train, y_train)
        best_model = model
        best_params = {}
        cv_score = scores.mean()
        cv_std = scores.std()

    elapsed = time.time() - start
    logger.info(f"    CV F1: {cv_score:.4f} ± {cv_std:.4f} ({elapsed:.1f}s)")
    if best_params:
        logger.info(f"    Best params: {best_params}")

    return {
        "name": name,
        "model": best_model,
        "best_params": best_params,
        "cv_score": cv_score,
        "cv_std": cv_std,
        "train_time": elapsed,
    }


def train_all_models(X_train, y_train) -> list[dict]:
    """
    Train all models and return results sorted by CV F1-score.

    Parameters
    ----------
    X_train, y_train : array-like
        Training data.

    Returns
    -------
    list[dict]
        List of result dicts, sorted by cv_score (descending).
    """
    logger.info("=" * 50)
    logger.info("MODEL TRAINING")
    logger.info("=" * 50)
    logger.info(f"  Cross-validation: {CV_FOLDS}-fold Stratified K-Fold")
    logger.info(f"  Scoring metric: F1-score")
    logger.info(f"  Random seed: {RANDOM_SEED}")
    logger.info("")

    models = _get_models()
    results = []

    for name, config in models.items():
        result = train_single_model(
            name=name,
            model=config["model"],
            params=config["params"],
            X_train=X_train,
            y_train=y_train,
            tune=config["tune"],
        )
        results.append(result)

    # Sort by CV score (descending)
    results.sort(key=lambda x: x["cv_score"], reverse=True)

    logger.info("")
    logger.info("  Model Rankings (by CV F1-score):")
    for i, r in enumerate(results, 1):
        logger.info(f"    {i}. {r['name']}: {r['cv_score']:.4f} ± {r['cv_std']:.4f}")

    best = results[0]
    logger.info(f"\n  ★ Best model: {best['name']} (F1={best['cv_score']:.4f})")

    return results
