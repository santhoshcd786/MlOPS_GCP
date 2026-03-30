"""
Feature Engineering Module
===========================
Generates customer behavioral features for purchase price prediction.
Used inside the Vertex AI Pipeline as a component.
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Category → price tier mapping ───────────────────────────────────────────
CATEGORY_PRICE_TIERS = {
    "Electronics":    "high",
    "Jewelry":        "high",
    "Appliances":     "high",
    "Furniture":      "high",
    "Clothing":       "medium",
    "Sports":         "medium",
    "Books":          "low",
    "Groceries":      "low",
    "Accessories":    "medium",
    "Beauty":         "medium",
}

LOYALTY_ORDER = {"Bronze": 0, "Silver": 1, "Gold": 2, "Platinum": 3}


def load_raw_data(input_path: str) -> pd.DataFrame:
    """Load raw transaction data from CSV or BigQuery export."""
    logger.info(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Validate required columns exist."""
    required_cols = [
        "customer_id", "transaction_date", "item_category",
        "purchase_price", "customer_age", "loyalty_tier",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    logger.info("Schema validation passed ✓")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main feature engineering pipeline.
    
    Creates:
      - Recency / Frequency / Monetary (RFM) features
      - Behavioral engagement features
      - Encoded categorical features
      - Target variable: highest_price_item_category
    """
    logger.info("Starting feature engineering ...")
    df = df.copy()

    # ── Date parsing ──────────────────────────────────────────────────────────
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    reference_date = df["transaction_date"].max()

    # ── RFM features (per customer) ───────────────────────────────────────────
    rfm = df.groupby("customer_id").agg(
        days_since_last_purchase=("transaction_date",
                                  lambda x: (reference_date - x.max()).days),
        purchase_frequency=("transaction_id", "count"),
        avg_purchase_value=("purchase_price", "mean"),
        total_spend=("purchase_price", "sum"),
        max_single_purchase=("purchase_price", "max"),
        std_purchase_value=("purchase_price", "std"),
    ).reset_index()
    rfm["std_purchase_value"] = rfm["std_purchase_value"].fillna(0)

    # ── Top purchased category per customer ───────────────────────────────────
    top_cat = (
        df.groupby(["customer_id", "item_category"])
        .size()
        .reset_index(name="cat_count")
        .sort_values("cat_count", ascending=False)
        .drop_duplicates("customer_id")[["customer_id", "item_category"]]
        .rename(columns={"item_category": "top_category"})
    )

    # ── Target: highest-priced item category bought by customer ───────────────
    target = (
        df.sort_values("purchase_price", ascending=False)
        .drop_duplicates("customer_id")[["customer_id", "item_category"]]
        .rename(columns={"item_category": "highest_price_item_category"})
    )

    # ── Static customer features (take latest record) ─────────────────────────
    customer_static = (
        df.sort_values("transaction_date", ascending=False)
        .drop_duplicates("customer_id")[[
            "customer_id", "customer_age", "loyalty_tier",
            "device_type", "browsing_time_mins", "cart_abandonment_rate",
        ]]
    )

    # ── Merge all features ────────────────────────────────────────────────────
    features = (
        customer_static
        .merge(rfm, on="customer_id")
        .merge(top_cat, on="customer_id")
        .merge(target, on="customer_id")
    )

    # ── Derived features ──────────────────────────────────────────────────────
    features["spend_per_visit"] = (
        features["total_spend"] / features["purchase_frequency"]
    )
    features["recency_bucket"] = pd.cut(
        features["days_since_last_purchase"],
        bins=[0, 7, 30, 90, 365, 9999],
        labels=["very_recent", "recent", "moderate", "old", "churned"],
    ).astype(str)
    features["loyalty_score"] = features["loyalty_tier"].map(LOYALTY_ORDER).fillna(0)
    features["price_tier"] = features["highest_price_item_category"].map(
        CATEGORY_PRICE_TIERS
    ).fillna("medium")

    # ── Encode categoricals ───────────────────────────────────────────────────
    cat_cols = ["top_category", "device_type", "recency_bucket", "loyalty_tier"]
    for col in cat_cols:
        le = LabelEncoder()
        features[f"{col}_enc"] = le.fit_transform(features[col].astype(str))

    # ── Encode target ─────────────────────────────────────────────────────────
    le_target = LabelEncoder()
    features["target_encoded"] = le_target.fit_transform(
        features["highest_price_item_category"]
    )

    logger.info(f"Feature engineering complete. Output shape: {features.shape}")
    return features, le_target


def get_feature_columns() -> list:
    """Return the list of feature columns used for training."""
    return [
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


def main(input_path: str, output_path: str) -> None:
    """Entry point for Vertex AI Pipeline component."""
    df_raw = load_raw_data(input_path)
    validate_schema(df_raw)
    features, label_encoder = engineer_features(df_raw)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)
    logger.info(f"Features saved to: {output_path}")

    # Save label encoder classes for serving
    classes_path = os.path.join(os.path.dirname(output_path), "label_classes.txt")
    with open(classes_path, "w") as f:
        f.write("\n".join(label_encoder.classes_))
    logger.info(f"Label classes saved to: {classes_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path",  required=True, help="Path to raw CSV data")
    parser.add_argument("--output-path", required=True, help="Path to save features CSV")
    args = parser.parse_args()
    main(args.input_path, args.output_path)
