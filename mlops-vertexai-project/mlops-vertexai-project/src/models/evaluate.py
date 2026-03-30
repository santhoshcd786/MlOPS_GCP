"""
Model Evaluation Module
========================
Evaluates trained model on holdout test set.
Outputs metrics JSON used by Vertex AI Pipeline gate
to decide whether to proceed with deployment.
"""

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "customer_age", "avg_purchase_value", "purchase_frequency",
    "days_since_last_purchase", "total_spend", "max_single_purchase",
    "std_purchase_value", "spend_per_visit", "browsing_time_mins",
    "cart_abandonment_rate", "loyalty_score", "top_category_enc",
    "device_type_enc", "recency_bucket_enc", "loyalty_tier_enc",
]
TARGET_COL = "target_encoded"


def load_model(model_dir: str) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier()
    model.load_model(f"{model_dir}/model.bst")
    logger.info("Model loaded ✓")
    return model


def load_test_data(features_path: str, test_size: float = 0.15):
    df = pd.read_csv(features_path)
    # Use same seed as training to get the same test split
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(
        df[FEATURE_COLS].values,
        df[TARGET_COL].values,
        test_size=test_size,
        random_state=42,
        stratify=df[TARGET_COL].values,
    )
    logger.info(f"Test set size: {len(X_test)}")
    return X_test, y_test


def evaluate(model, X_test, y_test, class_names: list = None) -> dict:
    """Compute all evaluation metrics."""
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1       = f1_score(y_test, y_pred, average="weighted")
    cm       = confusion_matrix(y_test, y_pred).tolist()
    report   = classification_report(
        y_test, y_pred,
        target_names=class_names,
        output_dict=True,
    )

    # ROC-AUC (multi-class OvR)
    try:
        roc_auc = roc_auc_score(
            y_test, y_pred_prob, multi_class="ovr", average="macro"
        )
    except Exception:
        roc_auc = None

    metrics = {
        "accuracy":          round(float(accuracy), 4),
        "f1_weighted":       round(float(f1), 4),
        "roc_auc_macro":     round(float(roc_auc), 4) if roc_auc else None,
        "confusion_matrix":  cm,
        "classification_report": report,
        "n_test_samples":    len(y_test),
    }

    logger.info(f"  Accuracy   : {accuracy:.4f}")
    logger.info(f"  F1 (wtd)   : {f1:.4f}")
    if roc_auc:
        logger.info(f"  ROC-AUC    : {roc_auc:.4f}")

    return metrics


def check_deployment_gate(metrics: dict, accuracy_threshold: float,
                           f1_threshold: float) -> bool:
    """Return True if model passes deployment quality gate."""
    passes = (
        metrics["accuracy"]    >= accuracy_threshold and
        metrics["f1_weighted"] >= f1_threshold
    )
    status = "PASSED ✓" if passes else "FAILED ✗"
    logger.info(
        f"Deployment gate [{status}] — "
        f"Accuracy: {metrics['accuracy']:.4f} (threshold: {accuracy_threshold}), "
        f"F1: {metrics['f1_weighted']:.4f} (threshold: {f1_threshold})"
    )
    return passes


def main(model_dir: str, features_path: str, metrics_output_path: str,
         accuracy_threshold: float, f1_threshold: float,
         label_classes_path: str = None) -> None:

    # Load class names
    class_names = None
    if label_classes_path and Path(label_classes_path).exists():
        with open(label_classes_path) as f:
            class_names = [line.strip() for line in f.readlines()]

    model = load_model(model_dir)
    X_test, y_test = load_test_data(features_path)
    metrics = evaluate(model, X_test, y_test, class_names)

    # Deployment gate decision
    deploy = check_deployment_gate(metrics, accuracy_threshold, f1_threshold)
    metrics["deploy"] = deploy
    metrics["accuracy_threshold"] = accuracy_threshold
    metrics["f1_threshold"]       = f1_threshold

    # Save metrics
    Path(metrics_output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to: {metrics_output_path}")

    if not deploy:
        raise SystemExit(
            "Model did not meet quality thresholds. Deployment aborted."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir",           required=True)
    parser.add_argument("--features-path",        required=True)
    parser.add_argument("--metrics-output-path",  required=True)
    parser.add_argument("--label-classes-path",   default=None)
    parser.add_argument("--accuracy-threshold",   type=float, default=0.80)
    parser.add_argument("--f1-threshold",         type=float, default=0.78)
    args = parser.parse_args()

    main(
        model_dir=args.model_dir,
        features_path=args.features_path,
        metrics_output_path=args.metrics_output_path,
        accuracy_threshold=args.accuracy_threshold,
        f1_threshold=args.f1_threshold,
        label_classes_path=args.label_classes_path,
    )
