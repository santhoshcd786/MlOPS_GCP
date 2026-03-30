"""
Prediction Serving Module
==========================
Custom predictor class for Vertex AI Endpoint.
Handles preprocessing, inference, and postprocessing
for the purchase price prediction model.
"""

import json
import logging
import os
import pickle
from typing import Any, Dict, List

import numpy as np
import xgboost as xgb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Feature order must match training
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

# Static encoding maps (match LabelEncoder output from training)
LOYALTY_SCORE_MAP = {"Bronze": 0, "Silver": 1, "Gold": 2, "Platinum": 3}
TOP_CATEGORY_ENC  = {
    "Accessories": 0, "Appliances": 1, "Beauty": 2, "Books": 3,
    "Clothing": 4, "Electronics": 5, "Furniture": 6,
    "Groceries": 7, "Jewelry": 8, "Sports": 9,
}
DEVICE_TYPE_ENC   = {"Desktop": 0, "Mobile": 1, "Tablet": 2}
RECENCY_BUCKET_ENC = {
    "churned": 0, "moderate": 1, "old": 2,
    "recent": 3, "very_recent": 4,
}
LOYALTY_TIER_ENC  = {"Bronze": 0, "Gold": 1, "Platinum": 2, "Silver": 3}

# Category labels (index → class name)
ITEM_CATEGORIES = [
    "Accessories", "Appliances", "Beauty", "Books",
    "Clothing", "Electronics", "Furniture",
    "Groceries", "Jewelry", "Sports",
]

# Price tiers by category
CATEGORY_PRICE_MAP = {
    "Electronics": {"tier": "high",   "avg_price": 850},
    "Jewelry":     {"tier": "high",   "avg_price": 650},
    "Appliances":  {"tier": "high",   "avg_price": 720},
    "Furniture":   {"tier": "high",   "avg_price": 900},
    "Clothing":    {"tier": "medium", "avg_price": 120},
    "Sports":      {"tier": "medium", "avg_price": 180},
    "Accessories": {"tier": "medium", "avg_price": 95},
    "Beauty":      {"tier": "medium", "avg_price": 75},
    "Books":       {"tier": "low",    "avg_price": 25},
    "Groceries":   {"tier": "low",    "avg_price": 45},
}


class PurchasePredictor:
    """
    Vertex AI Custom Predictor.
    
    Loaded by the serving container and called on each prediction request.
    """

    def __init__(self):
        self.model = None
        self._is_loaded = False

    def load(self, artifacts_uri: str) -> None:
        """Load model artifacts from GCS path."""
        model_path = os.path.join(artifacts_uri, "model.bst")
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        self._is_loaded = True
        logger.info(f"Model loaded from: {model_path}")

    def preprocess(self, instances: List[Dict[str, Any]]) -> np.ndarray:
        """
        Preprocess raw prediction instances into feature matrix.
        
        Expected input format (single instance):
        {
            "customer_age": 34,
            "avg_purchase_value": 120.5,
            "purchase_frequency": 8,
            "days_since_last_purchase": 14,
            "total_spend": 964.0,
            "max_single_purchase": 299.0,
            "std_purchase_value": 85.3,
            "browsing_time_mins": 22.5,
            "cart_abandonment_rate": 0.3,
            "loyalty_tier": "Gold",
            "top_category": "Electronics",
            "device_type": "Mobile",
            "recency_bucket": "recent"
        }
        """
        feature_matrix = []

        for instance in instances:
            # Encode categoricals
            loyalty_tier = instance.get("loyalty_tier", "Bronze")
            top_category = instance.get("top_category", "Clothing")
            device_type  = instance.get("device_type", "Mobile")
            recency_bucket = instance.get("recency_bucket", "moderate")

            spend = instance.get("total_spend", 0)
            freq  = max(instance.get("purchase_frequency", 1), 1)

            row = [
                instance.get("customer_age",              25),
                instance.get("avg_purchase_value",       100.0),
                freq,
                instance.get("days_since_last_purchase",  30),
                spend,
                instance.get("max_single_purchase",      200.0),
                instance.get("std_purchase_value",        50.0),
                spend / freq,                                       # spend_per_visit
                instance.get("browsing_time_mins",        15.0),
                instance.get("cart_abandonment_rate",      0.2),
                LOYALTY_SCORE_MAP.get(loyalty_tier, 0),            # loyalty_score
                TOP_CATEGORY_ENC.get(top_category, 0),             # top_category_enc
                DEVICE_TYPE_ENC.get(device_type, 1),               # device_type_enc
                RECENCY_BUCKET_ENC.get(recency_bucket, 1),         # recency_bucket_enc
                LOYALTY_TIER_ENC.get(loyalty_tier, 0),             # loyalty_tier_enc
            ]
            feature_matrix.append(row)

        return np.array(feature_matrix, dtype=np.float32)

    def predict(self, instances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run inference and return structured predictions."""
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        X = self.preprocess(instances)
        y_pred      = self.model.predict(X)
        y_pred_prob = self.model.predict_proba(X)

        predictions = []
        for i, (label_idx, probs) in enumerate(zip(y_pred, y_pred_prob)):
            predicted_category = ITEM_CATEGORIES[int(label_idx)]
            price_info = CATEGORY_PRICE_MAP.get(
                predicted_category, {"tier": "medium", "avg_price": 100}
            )

            # Top 3 likely categories
            top3_idx = np.argsort(probs)[::-1][:3]
            top3 = [
                {
                    "category":    ITEM_CATEGORIES[idx],
                    "probability": round(float(probs[idx]), 4),
                    "price_tier":  CATEGORY_PRICE_MAP.get(
                        ITEM_CATEGORIES[idx], {}
                    ).get("tier", "medium"),
                    "avg_price":   CATEGORY_PRICE_MAP.get(
                        ITEM_CATEGORIES[idx], {}
                    ).get("avg_price", 0),
                }
                for idx in top3_idx
            ]

            predictions.append({
                "predicted_category":    predicted_category,
                "confidence":            round(float(probs[label_idx]), 4),
                "price_tier":            price_info["tier"],
                "estimated_price":       price_info["avg_price"],
                "top_3_predictions":     top3,
                "recommendation":        self._build_recommendation(
                    predicted_category, price_info["tier"], instances[i]
                ),
            })

        return predictions

    @staticmethod
    def _build_recommendation(category: str, price_tier: str,
                               instance: Dict) -> str:
        loyalty = instance.get("loyalty_tier", "Bronze")
        freq    = instance.get("purchase_frequency", 1)

        if price_tier == "high" and loyalty in ("Gold", "Platinum"):
            return (
                f"Premium {category} recommendation — "
                f"High-value customer with {freq} purchases/month. "
                f"Offer exclusive early access or loyalty discount."
            )
        elif price_tier == "high":
            return (
                f"Upsell opportunity: {category} — "
                f"Customer shows interest in premium items. "
                f"Consider introductory offer to convert."
            )
        else:
            return (
                f"Personalized {category} suggestion — "
                f"Matches purchase history and browsing behavior."
            )


# ── Standalone test ───────────────────────────────────────────────────────────

def local_predict(model_dir: str, instances: List[Dict]) -> List[Dict]:
    """Helper for local testing without Vertex AI endpoint."""
    predictor = PurchasePredictor()
    predictor.load(model_dir)
    return predictor.predict(instances)


if __name__ == "__main__":
    # Quick local smoke test
    sample_instances = [
        {
            "customer_age": 34,
            "avg_purchase_value": 320.5,
            "purchase_frequency": 12,
            "days_since_last_purchase": 5,
            "total_spend": 3846.0,
            "max_single_purchase": 899.0,
            "std_purchase_value": 210.3,
            "browsing_time_mins": 45.0,
            "cart_abandonment_rate": 0.15,
            "loyalty_tier": "Gold",
            "top_category": "Electronics",
            "device_type": "Desktop",
            "recency_bucket": "very_recent",
        },
        {
            "customer_age": 22,
            "avg_purchase_value": 45.0,
            "purchase_frequency": 3,
            "days_since_last_purchase": 60,
            "total_spend": 135.0,
            "max_single_purchase": 75.0,
            "std_purchase_value": 15.0,
            "browsing_time_mins": 8.0,
            "cart_abandonment_rate": 0.6,
            "loyalty_tier": "Bronze",
            "top_category": "Books",
            "device_type": "Mobile",
            "recency_bucket": "moderate",
        },
    ]

    print("Sample prediction output (without loaded model — schema only):")
    predictor = PurchasePredictor()
    X = predictor.preprocess(sample_instances)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Feature columns: {FEATURE_COLS}")
    print("\nTo run full inference, load a trained model first.")
