"""
Unit Tests — Feature Engineering & Model
==========================================
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.features.feature_engineering import (
    engineer_features,
    get_feature_columns,
    validate_schema,
)
from src.serving.predictor import PurchasePredictor, FEATURE_COLS


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_raw_df():
    """Minimal synthetic transaction DataFrame for testing."""
    np.random.seed(42)
    n = 200
    customers = [f"CUST_{i:04d}" for i in range(1, 21)]

    data = {
        "customer_id":           np.random.choice(customers, n),
        "transaction_id":        [f"TXN_{i:05d}" for i in range(n)],
        "transaction_date":      pd.date_range("2024-01-01", periods=n, freq="2D").strftime("%Y-%m-%d"),
        "item_category":         np.random.choice(
            ["Electronics", "Clothing", "Books", "Sports", "Jewelry",
             "Furniture", "Groceries", "Beauty", "Appliances", "Accessories"], n
        ),
        "purchase_price":        np.random.uniform(20, 1000, n).round(2),
        "customer_age":          np.random.randint(18, 70, n),
        "loyalty_tier":          np.random.choice(
            ["Bronze", "Silver", "Gold", "Platinum"], n,
            p=[0.45, 0.30, 0.18, 0.07]
        ),
        "device_type":           np.random.choice(["Mobile", "Desktop", "Tablet"], n),
        "browsing_time_mins":    np.random.exponential(20, n).round(1),
        "cart_abandonment_rate": np.random.beta(2, 5, n).round(2),
    }
    return pd.DataFrame(data)


# ── Schema Validation Tests ───────────────────────────────────────────────────

class TestSchemaValidation:
    def test_valid_schema_passes(self, sample_raw_df):
        """Should not raise for valid data."""
        validate_schema(sample_raw_df)  # no exception

    def test_missing_column_raises(self, sample_raw_df):
        """Should raise ValueError for missing column."""
        df_bad = sample_raw_df.drop(columns=["customer_id"])
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_schema(df_bad)

    def test_all_required_columns_checked(self, sample_raw_df):
        """Each required column, when removed, should trigger error."""
        required = ["customer_id", "transaction_date", "item_category", "purchase_price"]
        for col in required:
            df_bad = sample_raw_df.drop(columns=[col])
            with pytest.raises(ValueError):
                validate_schema(df_bad)


# ── Feature Engineering Tests ─────────────────────────────────────────────────

class TestFeatureEngineering:
    def test_output_has_expected_feature_columns(self, sample_raw_df):
        features, _ = engineer_features(sample_raw_df)
        expected = get_feature_columns()
        for col in expected:
            assert col in features.columns, f"Missing feature column: {col}"

    def test_one_row_per_customer(self, sample_raw_df):
        features, _ = engineer_features(sample_raw_df)
        n_customers = sample_raw_df["customer_id"].nunique()
        assert len(features) == n_customers

    def test_no_nulls_in_numeric_features(self, sample_raw_df):
        features, _ = engineer_features(sample_raw_df)
        numeric_cols = get_feature_columns()
        nulls = features[numeric_cols].isnull().sum()
        assert nulls.sum() == 0, f"Found nulls: {nulls[nulls > 0]}"

    def test_target_column_present(self, sample_raw_df):
        features, _ = engineer_features(sample_raw_df)
        assert "highest_price_item_category" in features.columns
        assert "target_encoded" in features.columns

    def test_encoded_values_are_integers(self, sample_raw_df):
        features, _ = engineer_features(sample_raw_df)
        for col in ["top_category_enc", "device_type_enc",
                    "recency_bucket_enc", "loyalty_tier_enc"]:
            assert pd.api.types.is_integer_dtype(features[col]), \
                f"{col} should be integer"

    def test_loyalty_score_in_valid_range(self, sample_raw_df):
        features, _ = engineer_features(sample_raw_df)
        assert features["loyalty_score"].between(0, 3).all()

    def test_spend_per_visit_non_negative(self, sample_raw_df):
        features, _ = engineer_features(sample_raw_df)
        assert (features["spend_per_visit"] >= 0).all()

    def test_label_encoder_classes_match_categories(self, sample_raw_df):
        features, le = engineer_features(sample_raw_df)
        assert len(le.classes_) > 0
        # All encoded values should map back to a valid class
        for enc_val in features["target_encoded"].unique():
            assert 0 <= enc_val < len(le.classes_)


# ── Feature Columns Tests ─────────────────────────────────────────────────────

class TestFeatureColumns:
    def test_feature_columns_returns_list(self):
        cols = get_feature_columns()
        assert isinstance(cols, list)

    def test_feature_columns_not_empty(self):
        cols = get_feature_columns()
        assert len(cols) > 0

    def test_feature_columns_no_duplicates(self):
        cols = get_feature_columns()
        assert len(cols) == len(set(cols))


# ── Predictor Preprocessing Tests ────────────────────────────────────────────

class TestPredictorPreprocessing:
    def test_preprocess_returns_correct_shape(self):
        predictor = PurchasePredictor()
        instances = [
            {
                "customer_age": 34,
                "avg_purchase_value": 200.0,
                "purchase_frequency": 5,
                "days_since_last_purchase": 10,
                "total_spend": 1000.0,
                "max_single_purchase": 400.0,
                "std_purchase_value": 100.0,
                "browsing_time_mins": 20.0,
                "cart_abandonment_rate": 0.2,
                "loyalty_tier": "Gold",
                "top_category": "Electronics",
                "device_type": "Mobile",
                "recency_bucket": "recent",
            }
        ]
        X = predictor.preprocess(instances)
        assert X.shape == (1, len(FEATURE_COLS)), \
            f"Expected shape (1, {len(FEATURE_COLS)}), got {X.shape}"

    def test_preprocess_handles_multiple_instances(self):
        predictor = PurchasePredictor()
        instances = [
            {"customer_age": 25, "loyalty_tier": "Bronze",
             "top_category": "Books", "device_type": "Mobile",
             "recency_bucket": "moderate"},
            {"customer_age": 45, "loyalty_tier": "Platinum",
             "top_category": "Jewelry", "device_type": "Desktop",
             "recency_bucket": "very_recent"},
        ]
        X = predictor.preprocess(instances)
        assert X.shape[0] == 2

    def test_preprocess_handles_missing_fields_with_defaults(self):
        predictor = PurchasePredictor()
        instances = [{}]  # empty instance → all defaults
        X = predictor.preprocess(instances)
        assert X.shape == (1, len(FEATURE_COLS))
        assert not np.isnan(X).any()

    def test_preprocess_unknown_category_does_not_crash(self):
        predictor = PurchasePredictor()
        instances = [{"top_category": "UNKNOWN_CAT", "device_type": "VR_Headset"}]
        X = predictor.preprocess(instances)  # should not raise
        assert X.shape == (1, len(FEATURE_COLS))
