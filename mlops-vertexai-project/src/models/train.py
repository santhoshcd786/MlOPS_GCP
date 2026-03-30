"""
Model Training Module
======================
Trains an XGBoost classifier to predict the highest-priced item
category a customer is most likely to purchase.

Designed to run as a Vertex AI Training Job.
"""

import argparse
import json
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

# Optional: Google Cloud integration
try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Feature columns (must match feature_engineering.py) ──────────────────────
FEATURE_COLS = [
    "customer_age",
    "avg_purchase_value",
    "purchase_frequency",
    "days_since_last_purchase",
    "total_spend",
    "max_single_purchase",
    "std_purchase_value",
    "spend_per_visit",
    "browsing_time_mins",
    "cart_abandonment_rate",
    "loyalty_score",
    "top_category_enc",
    "device_type_enc",
    "recency_bucket_enc",
    "loyalty_tier_enc",
]
TARGET_COL = "target_encoded"


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_features(features_path: str) -> pd.DataFrame:
    logger.info(f"Loading features from: {features_path}")
    df = pd.read_csv(features_path)
    logger.info(f"Dataset shape: {df.shape}")
    return df


def prepare_splits(df: pd.DataFrame, test_size: float = 0.15,
                   val_size: float = 0.15, random_state: int = 42):
    """Split into train / validation / test sets (stratified)."""
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    relative_val_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=relative_val_size,
        random_state=random_state, stratify=y_trainval
    )

    logger.info(
        f"Split sizes → Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ── Model Definition ──────────────────────────────────────────────────────────

def build_model(params: dict, n_classes: int) -> xgb.XGBClassifier:
    """Build XGBoost classifier with given hyperparameters."""
    default_params = {
        "objective":        "multi:softprob",
        "num_class":        n_classes,
        "n_estimators":     300,
        "max_depth":        6,
        "learning_rate":    0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "gamma":            0.1,
        "reg_alpha":        0.1,
        "reg_lambda":       1.0,
        "use_label_encoder": False,
        "eval_metric":      "mlogloss",
        "random_state":     42,
        "n_jobs":           -1,
        "verbosity":        1,
    }
    default_params.update(params)
    return xgb.XGBClassifier(**default_params)


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(X_train, y_train, X_val, y_val, hyperparams: dict,
                n_classes: int) -> xgb.XGBClassifier:
    """Train XGBoost with early stopping on validation set."""
    logger.info("Training model ...")
    model = build_model(hyperparams, n_classes)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=30,
        verbose=50,
    )

    best_iter = model.best_iteration
    logger.info(f"Best iteration: {best_iter}")
    return model


# ── Cross-validation ──────────────────────────────────────────────────────────

def cross_validate_model(X_train, y_train, hyperparams: dict,
                         n_classes: int, cv: int = 5) -> dict:
    """Run stratified k-fold CV and return mean/std scores."""
    logger.info(f"Running {cv}-fold cross-validation ...")
    model = build_model(hyperparams, n_classes)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    acc_scores = cross_val_score(model, X_train, y_train, cv=skf,
                                 scoring="accuracy", n_jobs=-1)
    f1_scores  = cross_val_score(model, X_train, y_train, cv=skf,
                                 scoring="f1_weighted", n_jobs=-1)

    cv_results = {
        "cv_accuracy_mean": float(acc_scores.mean()),
        "cv_accuracy_std":  float(acc_scores.std()),
        "cv_f1_mean":       float(f1_scores.mean()),
        "cv_f1_std":        float(f1_scores.std()),
    }
    logger.info(
        f"CV Accuracy: {cv_results['cv_accuracy_mean']:.4f} "
        f"± {cv_results['cv_accuracy_std']:.4f}"
    )
    return cv_results


# ── Feature Importance ────────────────────────────────────────────────────────

def get_feature_importance(model: xgb.XGBClassifier) -> dict:
    importance = model.feature_importances_
    fi_dict = {col: float(imp) for col, imp in zip(FEATURE_COLS, importance)}
    sorted_fi = dict(sorted(fi_dict.items(), key=lambda x: x[1], reverse=True))
    logger.info("Top 5 features:")
    for feat, score in list(sorted_fi.items())[:5]:
        logger.info(f"  {feat}: {score:.4f}")
    return sorted_fi


# ── Save Artifacts ────────────────────────────────────────────────────────────

def save_model(model: xgb.XGBClassifier, output_dir: str,
               metadata: dict) -> None:
    """Save model, metadata, and feature list for Vertex AI."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # XGBoost native format (preferred for Vertex AI)
    model_path = os.path.join(output_dir, "model.bst")
    model.save_model(model_path)
    logger.info(f"Model saved: {model_path}")

    # Also save as pickle for sklearn compatibility
    pkl_path = os.path.join(output_dir, "model.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(model, f)

    # Save metadata
    meta_path = os.path.join(output_dir, "training_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved: {meta_path}")

    # Save feature columns list
    feat_path = os.path.join(output_dir, "feature_columns.json")
    with open(feat_path, "w") as f:
        json.dump(FEATURE_COLS, f, indent=2)
    logger.info(f"Feature columns saved: {feat_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(features_path: str, model_output_dir: str,
         hyperparams: dict, test_size: float, val_size: float) -> None:

    # Load data
    df = load_features(features_path)
    n_classes = df[TARGET_COL].nunique()
    logger.info(f"Number of classes (item categories): {n_classes}")

    # Split
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_splits(
        df, test_size=test_size, val_size=val_size
    )

    # Cross-validate
    cv_results = cross_validate_model(X_train, y_train, hyperparams, n_classes)

    # Train final model
    model = train_model(X_train, y_train, X_val, y_val, hyperparams, n_classes)

    # Feature importance
    feature_importance = get_feature_importance(model)

    # Compile training metadata
    metadata = {
        "feature_columns":   FEATURE_COLS,
        "target_column":     TARGET_COL,
        "n_classes":         n_classes,
        "train_samples":     len(X_train),
        "val_samples":       len(X_val),
        "test_samples":      len(X_test),
        "hyperparameters":   hyperparams,
        "cv_results":        cv_results,
        "feature_importance": feature_importance,
        "best_iteration":    int(model.best_iteration),
    }

    # Save artifacts
    save_model(model, model_output_dir, metadata)
    logger.info("Training complete ✓")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train purchase prediction model")
    parser.add_argument("--features-path",    required=True)
    parser.add_argument("--model-output-dir", required=True)
    parser.add_argument("--n-estimators",     type=int,   default=300)
    parser.add_argument("--max-depth",        type=int,   default=6)
    parser.add_argument("--learning-rate",    type=float, default=0.05)
    parser.add_argument("--subsample",        type=float, default=0.8)
    parser.add_argument("--test-size",        type=float, default=0.15)
    parser.add_argument("--val-size",         type=float, default=0.15)
    args = parser.parse_args()

    hyperparams = {
        "n_estimators":     args.n_estimators,
        "max_depth":        args.max_depth,
        "learning_rate":    args.learning_rate,
        "subsample":        args.subsample,
        "colsample_bytree": 0.8,
    }

    main(
        features_path=args.features_path,
        model_output_dir=args.model_output_dir,
        hyperparams=hyperparams,
        test_size=args.test_size,
        val_size=args.val_size,
    )
