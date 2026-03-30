"""
generate_sample_data.py
========================
Generates synthetic e-commerce transaction data for development/testing.
Saves to data/sample_data.csv — upload to BigQuery with setup_gcp.sh.
"""

import logging
import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

random.seed(42)
np.random.seed(42)

# ── Configuration ─────────────────────────────────────────────────────────────
N_CUSTOMERS    = 5_000
MAX_TRANS      = 30          # max transactions per customer
OUTPUT_PATH    = "data/sample_data.csv"

CATEGORIES = [
    "Electronics", "Clothing", "Books", "Groceries",
    "Sports", "Beauty", "Jewelry", "Furniture",
    "Appliances", "Accessories",
]

CATEGORY_PRICE_RANGES = {
    "Electronics": (200, 1500),
    "Clothing":    (30,  300),
    "Books":       (10,  50),
    "Groceries":   (15,  120),
    "Sports":      (50,  500),
    "Beauty":      (20,  150),
    "Jewelry":     (100, 1200),
    "Furniture":   (200, 2000),
    "Appliances":  (150, 1800),
    "Accessories": (25,  250),
}

LOYALTY_TIERS  = ["Bronze", "Silver", "Gold", "Platinum"]
DEVICE_TYPES   = ["Mobile", "Desktop", "Tablet"]


def random_date(days_back: int = 365) -> datetime:
    return datetime.now() - timedelta(days=random.randint(0, days_back))


def generate_customer_profile(customer_id: int) -> dict:
    age   = random.randint(18, 75)
    tier  = np.random.choice(
        LOYALTY_TIERS, p=[0.45, 0.30, 0.18, 0.07]
    )
    # Higher loyalty → more high-value categories
    if tier in ("Gold", "Platinum"):
        preferred_cats = np.random.choice(
            ["Electronics", "Jewelry", "Appliances", "Furniture", "Clothing"],
            size=3, replace=False
        )
    elif tier == "Silver":
        preferred_cats = np.random.choice(
            ["Clothing", "Sports", "Beauty", "Accessories", "Electronics"],
            size=3, replace=False
        )
    else:
        preferred_cats = np.random.choice(
            ["Books", "Groceries", "Clothing", "Beauty", "Accessories"],
            size=3, replace=False
        )

    return {
        "customer_id":           f"CUST_{customer_id:06d}",
        "customer_age":          age,
        "loyalty_tier":          tier,
        "device_type":           np.random.choice(
            DEVICE_TYPES, p=[0.55, 0.35, 0.10]
        ),
        "browsing_time_mins":    round(np.random.exponential(scale=20) + 3, 1),
        "cart_abandonment_rate": round(np.random.beta(2, 5), 2),
        "preferred_categories":  list(preferred_cats),
    }


def generate_transactions(customer: dict) -> list:
    n_trans = random.randint(1, MAX_TRANS)
    transactions = []

    for t in range(n_trans):
        # Bias toward preferred categories
        if random.random() < 0.65:
            category = random.choice(customer["preferred_categories"])
        else:
            category = random.choice(CATEGORIES)

        price_range = CATEGORY_PRICE_RANGES[category]
        price       = round(random.uniform(*price_range), 2)

        # Add noise to browsing time per transaction
        browsing = max(
            1.0,
            round(customer["browsing_time_mins"] + np.random.normal(0, 5), 1)
        )

        transactions.append({
            "customer_id":          customer["customer_id"],
            "transaction_id":       f"TXN_{customer['customer_id']}_{t:03d}",
            "transaction_date":     random_date(365).strftime("%Y-%m-%d"),
            "item_category":        category,
            "purchase_price":       price,
            "customer_age":         customer["customer_age"],
            "loyalty_tier":         customer["loyalty_tier"],
            "device_type":          customer["device_type"],
            "browsing_time_mins":   browsing,
            "cart_abandonment_rate": customer["cart_abandonment_rate"],
        })

    return transactions


def main():
    os.makedirs("data", exist_ok=True)
    logger.info(f"Generating data for {N_CUSTOMERS:,} customers ...")

    all_transactions = []
    for cid in range(1, N_CUSTOMERS + 1):
        customer = generate_customer_profile(cid)
        txns     = generate_transactions(customer)
        all_transactions.extend(txns)

        if cid % 1000 == 0:
            logger.info(f"  Generated {cid:,} / {N_CUSTOMERS:,} customers ...")

    df = pd.DataFrame(all_transactions)
    df = df.sort_values("transaction_date").reset_index(drop=True)

    logger.info(f"Total transactions: {len(df):,}")
    logger.info(f"Unique customers  : {df['customer_id'].nunique():,}")
    logger.info(f"Category distribution:\n{df['item_category'].value_counts()}")
    logger.info(f"Loyalty tier distribution:\n{df['loyalty_tier'].value_counts()}")

    df.to_csv(OUTPUT_PATH, index=False)
    logger.info(f"Sample data saved to: {OUTPUT_PATH}")
    logger.info(
        "Upload to BigQuery with:\n"
        "  bash scripts/setup_gcp.sh"
    )


if __name__ == "__main__":
    main()
