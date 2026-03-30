"""
Vertex AI Training Pipeline
============================
End-to-end KFP v2 pipeline for:
  1. Data ingestion from BigQuery
  2. Feature engineering
  3. Model training
  4. Model evaluation + deployment gate
  5. Model registration in Vertex AI Model Registry
  6. Deployment to Vertex AI Endpoint

Run via: python scripts/run_pipeline.py
"""

import os
from typing import NamedTuple

import yaml
from google.cloud import aiplatform
from kfp import compiler, dsl
from kfp.dsl import (
    Artifact,
    Dataset,
    Input,
    Metrics,
    Model,
    Output,
    component,
    pipeline,
)


# ── Load config ───────────────────────────────────────────────────────────────

def load_config(config_path: str = "configs/pipeline_config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ══════════════════════════════════════════════════════════════════════════════
# COMPONENT 1 — Data Ingestion & Validation
# ══════════════════════════════════════════════════════════════════════════════

@component(
    base_image="python:3.10-slim",
    packages_to_install=["google-cloud-bigquery", "pandas", "pyarrow", "db-dtypes"],
)
def ingest_data_from_bigquery(
    project_id:    str,
    bq_dataset:    str,
    bq_table:      str,
    query_limit:   int,
    output_dataset: Output[Dataset],
) -> None:
    """Pull transaction data from BigQuery and save as CSV."""
    import logging
    from google.cloud import bigquery

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    client = bigquery.Client(project=project_id)
    query  = f"""
        SELECT
            customer_id,
            transaction_id,
            transaction_date,
            item_category,
            purchase_price,
            customer_age,
            loyalty_tier,
            device_type,
            browsing_time_mins,
            cart_abandonment_rate
        FROM `{project_id}.{bq_dataset}.{bq_table}`
        WHERE transaction_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 365 DAY)
        LIMIT {query_limit}
    """
    logger.info(f"Running BigQuery query (limit {query_limit}) ...")
    df = client.query(query).to_dataframe()
    logger.info(f"Fetched {len(df):,} rows")

    # Basic validation
    assert len(df) > 0, "No data returned from BigQuery!"
    assert "customer_id" in df.columns

    df.to_csv(output_dataset.path, index=False)
    logger.info(f"Data saved: {output_dataset.path}")


# ══════════════════════════════════════════════════════════════════════════════
# COMPONENT 2 — Feature Engineering
# ══════════════════════════════════════════════════════════════════════════════

@component(
    base_image="python:3.10-slim",
    packages_to_install=["pandas", "scikit-learn", "numpy"],
)
def feature_engineering_component(
    input_dataset:   Input[Dataset],
    output_features: Output[Dataset],
) -> None:
    """Run feature engineering pipeline."""
    import sys
    sys.path.insert(0, "/")

    import logging
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    from pathlib import Path

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # ── inline feature engineering (same logic as feature_engineering.py) ──
    df = pd.read_csv(input_dataset.path)
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    ref = df["transaction_date"].max()

    rfm = df.groupby("customer_id").agg(
        days_since_last_purchase=("transaction_date", lambda x: (ref - x.max()).days),
        purchase_frequency=("transaction_id", "count"),
        avg_purchase_value=("purchase_price", "mean"),
        total_spend=("purchase_price", "sum"),
        max_single_purchase=("purchase_price", "max"),
        std_purchase_value=("purchase_price", "std"),
    ).reset_index().fillna(0)

    top_cat = (df.groupby(["customer_id","item_category"]).size()
               .reset_index(name="n")
               .sort_values("n", ascending=False)
               .drop_duplicates("customer_id")[["customer_id","item_category"]]
               .rename(columns={"item_category":"top_category"}))

    target = (df.sort_values("purchase_price", ascending=False)
              .drop_duplicates("customer_id")[["customer_id","item_category"]]
              .rename(columns={"item_category":"highest_price_item_category"}))

    static = (df.sort_values("transaction_date", ascending=False)
              .drop_duplicates("customer_id")[[
                  "customer_id","customer_age","loyalty_tier",
                  "device_type","browsing_time_mins","cart_abandonment_rate"]])

    features = (static.merge(rfm, on="customer_id")
                      .merge(top_cat, on="customer_id")
                      .merge(target, on="customer_id"))

    features["spend_per_visit"] = features["total_spend"] / features["purchase_frequency"]

    loyalty_order = {"Bronze":0,"Silver":1,"Gold":2,"Platinum":3}
    features["loyalty_score"] = features["loyalty_tier"].map(loyalty_order).fillna(0)

    features["recency_bucket"] = pd.cut(
        features["days_since_last_purchase"],
        bins=[0,7,30,90,365,9999],
        labels=["very_recent","recent","moderate","old","churned"]
    ).astype(str)

    for col in ["top_category","device_type","recency_bucket","loyalty_tier"]:
        le = LabelEncoder()
        features[f"{col}_enc"] = le.fit_transform(features[col].astype(str))

    le_t = LabelEncoder()
    features["target_encoded"] = le_t.fit_transform(features["highest_price_item_category"])

    features.to_csv(output_features.path, index=False)
    logger.info(f"Features saved: {output_features.path} — shape {features.shape}")


# ══════════════════════════════════════════════════════════════════════════════
# COMPONENT 3 — Model Training
# ══════════════════════════════════════════════════════════════════════════════

@component(
    base_image="python:3.10-slim",
    packages_to_install=["xgboost==1.7.6", "scikit-learn", "pandas", "numpy"],
)
def train_model_component(
    input_features: Input[Dataset],
    model_artifact: Output[Model],
    metrics:        Output[Metrics],
    n_estimators:   int   = 300,
    max_depth:      int   = 6,
    learning_rate:  float = 0.05,
    subsample:      float = 0.8,
    test_size:      float = 0.15,
    val_size:       float = 0.15,
) -> None:
    """Train XGBoost model and log metrics."""
    import json, logging, os, pickle
    import numpy as np
    import pandas as pd
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    FEATURE_COLS = [
        "customer_age","avg_purchase_value","purchase_frequency",
        "days_since_last_purchase","total_spend","max_single_purchase",
        "std_purchase_value","spend_per_visit","browsing_time_mins",
        "cart_abandonment_rate","loyalty_score","top_category_enc",
        "device_type_enc","recency_bucket_enc","loyalty_tier_enc",
    ]

    df = pd.read_csv(input_features.path)
    X, y = df[FEATURE_COLS].values, df["target_encoded"].values
    n_classes = len(np.unique(y))

    X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=test_size,
                                                    random_state=42, stratify=y)
    rv = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=rv,
                                                        random_state=42, stratify=y_tv)

    model = xgb.XGBClassifier(
        objective="multi:softprob", num_class=n_classes,
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate, subsample=subsample,
        colsample_bytree=0.8, eval_metric="mlogloss",
        use_label_encoder=False, random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
              early_stopping_rounds=30, verbose=50)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1       = f1_score(y_test, y_pred, average="weighted")

    logger.info(f"Test Accuracy: {accuracy:.4f} | F1: {f1:.4f}")

    # Log to Vertex AI Metrics
    metrics.log_metric("accuracy",   round(accuracy, 4))
    metrics.log_metric("f1_weighted", round(f1, 4))
    metrics.log_metric("n_classes",  n_classes)

    # Save model
    os.makedirs(model_artifact.path, exist_ok=True)
    model.save_model(os.path.join(model_artifact.path, "model.bst"))
    with open(os.path.join(model_artifact.path, "model.pkl"), "wb") as f:
        pickle.dump(model, f)

    meta = {
        "accuracy": accuracy, "f1_weighted": f1,
        "n_classes": n_classes, "best_iteration": int(model.best_iteration),
        "feature_columns": FEATURE_COLS,
    }
    with open(os.path.join(model_artifact.path, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Model saved: {model_artifact.path}")


# ══════════════════════════════════════════════════════════════════════════════
# COMPONENT 4 — Evaluate & Deployment Gate
# ══════════════════════════════════════════════════════════════════════════════

@component(
    base_image="python:3.10-slim",
    packages_to_install=["xgboost==1.7.6", "scikit-learn", "pandas"],
)
def evaluate_and_gate(
    model_artifact:       Input[Model],
    features:             Input[Dataset],
    accuracy_threshold:   float,
    f1_threshold:         float,
) -> NamedTuple("Outputs", [("deploy", bool)]):
    """Check evaluation metrics and decide whether to deploy."""
    import json, logging, pickle
    import numpy as np
    import pandas as pd
    import xgboost as xgb
    from collections import namedtuple
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    FEATURE_COLS = [
        "customer_age","avg_purchase_value","purchase_frequency",
        "days_since_last_purchase","total_spend","max_single_purchase",
        "std_purchase_value","spend_per_visit","browsing_time_mins",
        "cart_abandonment_rate","loyalty_score","top_category_enc",
        "device_type_enc","recency_bucket_enc","loyalty_tier_enc",
    ]

    model = xgb.XGBClassifier()
    import os
    model.load_model(os.path.join(model_artifact.path, "model.bst"))

    df = pd.read_csv(features.path)
    X, y = df[FEATURE_COLS].values, df["target_encoded"].values
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.15,
                                             random_state=42, stratify=y)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1       = f1_score(y_test, y_pred, average="weighted")

    deploy = bool(accuracy >= accuracy_threshold and f1 >= f1_threshold)
    status = "DEPLOY ✓" if deploy else "REJECT ✗"
    logger.info(f"Gate [{status}] acc={accuracy:.4f} f1={f1:.4f}")

    Output = namedtuple("Outputs", ["deploy"])
    return Output(deploy=deploy)


# ══════════════════════════════════════════════════════════════════════════════
# COMPONENT 5 — Register Model in Vertex AI Model Registry
# ══════════════════════════════════════════════════════════════════════════════

@component(
    base_image="python:3.10-slim",
    packages_to_install=["google-cloud-aiplatform"],
)
def register_model(
    model_artifact:     Input[Model],
    project_id:         str,
    region:             str,
    model_display_name: str,
    staging_bucket:     str,
) -> NamedTuple("Outputs", [("model_resource_name", str)]):
    """Upload and register model in Vertex AI Model Registry."""
    import logging
    from collections import namedtuple
    from google.cloud import aiplatform

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    aiplatform.init(project=project_id, location=region,
                    staging_bucket=staging_bucket)

    model = aiplatform.Model.upload(
        display_name=model_display_name,
        artifact_uri=model_artifact.path,
        serving_container_image_uri=(
            "us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-6:latest"
        ),
        description="Customer purchase price prediction model",
        labels={"framework": "xgboost", "task": "purchase-prediction"},
    )
    logger.info(f"Model registered: {model.resource_name}")

    Output = namedtuple("Outputs", ["model_resource_name"])
    return Output(model_resource_name=model.resource_name)


# ══════════════════════════════════════════════════════════════════════════════
# COMPONENT 6 — Deploy to Vertex AI Endpoint
# ══════════════════════════════════════════════════════════════════════════════

@component(
    base_image="python:3.10-slim",
    packages_to_install=["google-cloud-aiplatform"],
)
def deploy_model_to_endpoint(
    model_resource_name: str,
    project_id:          str,
    region:              str,
    endpoint_display_name: str,
    machine_type:        str = "n1-standard-2",
    min_replicas:        int = 1,
    max_replicas:        int = 3,
) -> None:
    """Deploy registered model to a Vertex AI Endpoint."""
    import logging
    from google.cloud import aiplatform

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    aiplatform.init(project=project_id, location=region)

    # Get or create endpoint
    endpoints = aiplatform.Endpoint.list(
        filter=f'display_name="{endpoint_display_name}"',
        project=project_id, location=region,
    )
    endpoint = endpoints[0] if endpoints else aiplatform.Endpoint.create(
        display_name=endpoint_display_name,
        project=project_id, location=region,
    )
    logger.info(f"Using endpoint: {endpoint.resource_name}")

    model = aiplatform.Model(model_resource_name)
    model.deploy(
        endpoint=endpoint,
        deployed_model_display_name=f"{endpoint_display_name}-deployed",
        machine_type=machine_type,
        min_replica_count=min_replicas,
        max_replica_count=max_replicas,
        traffic_split={"0": 100},
    )
    logger.info(f"Model deployed to: {endpoint.resource_name}")


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE DEFINITION
# ══════════════════════════════════════════════════════════════════════════════

@pipeline(
    name="purchase-price-prediction-pipeline",
    description="End-to-end MLOps pipeline for customer purchase price prediction",
)
def purchase_prediction_pipeline(
    project_id:           str,
    region:               str,
    bq_dataset:           str,
    bq_table:             str,
    staging_bucket:       str,
    query_limit:          int   = 100000,
    n_estimators:         int   = 300,
    max_depth:            int   = 6,
    learning_rate:        float = 0.05,
    subsample:            float = 0.8,
    accuracy_threshold:   float = 0.80,
    f1_threshold:         float = 0.78,
    model_display_name:   str   = "purchase-predictor",
    endpoint_display_name: str  = "purchase-predictor-endpoint",
    machine_type:         str   = "n1-standard-2",
):
    # Step 1 — Ingest
    ingest_task = ingest_data_from_bigquery(
        project_id=project_id, bq_dataset=bq_dataset,
        bq_table=bq_table, query_limit=query_limit,
    )

    # Step 2 — Feature engineering
    feat_task = feature_engineering_component(
        input_dataset=ingest_task.outputs["output_dataset"]
    )

    # Step 3 — Train
    train_task = train_model_component(
        input_features=feat_task.outputs["output_features"],
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate, subsample=subsample,
    )

    # Step 4 — Evaluate & gate
    gate_task = evaluate_and_gate(
        model_artifact=train_task.outputs["model_artifact"],
        features=feat_task.outputs["output_features"],
        accuracy_threshold=accuracy_threshold,
        f1_threshold=f1_threshold,
    )

    # Step 5 — Register (only if gate passes)
    with dsl.If(gate_task.outputs["deploy"] == True, name="deployment-gate"):
        register_task = register_model(
            model_artifact=train_task.outputs["model_artifact"],
            project_id=project_id, region=region,
            model_display_name=model_display_name,
            staging_bucket=staging_bucket,
        )

        # Step 6 — Deploy
        deploy_model_to_endpoint(
            model_resource_name=register_task.outputs["model_resource_name"],
            project_id=project_id, region=region,
            endpoint_display_name=endpoint_display_name,
            machine_type=machine_type,
        )


# ══════════════════════════════════════════════════════════════════════════════
# COMPILE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=purchase_prediction_pipeline,
        package_path="purchase_prediction_pipeline.yaml",
    )
    print("Pipeline compiled → purchase_prediction_pipeline.yaml")
