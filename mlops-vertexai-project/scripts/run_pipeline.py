"""
run_pipeline.py
================
Submits the Vertex AI KFP pipeline job.
Usage:
    python scripts/run_pipeline.py --config configs/pipeline_config.yaml
"""

import argparse
import logging
import os
from datetime import datetime

import yaml
from google.cloud import aiplatform

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def compile_pipeline() -> str:
    """Compile the KFP pipeline and return path to YAML."""
    import subprocess
    compiled_path = "/tmp/purchase_prediction_pipeline.yaml"
    result = subprocess.run(
        ["python", "src/pipelines/training_pipeline.py"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Pipeline compilation failed:\n{result.stderr}")
    logger.info("Pipeline compiled ✓")
    return compiled_path


def run_pipeline(config: dict) -> None:
    gcp  = config["gcp"]
    pipe = config["pipeline"]
    tr   = config["training"]
    hp   = tr["hyperparameters"]
    ev   = config["evaluation"]
    srv  = config["serving"]

    # Init Vertex AI
    aiplatform.init(
        project=gcp["project_id"],
        location=gcp["region"],
        staging_bucket=gcp["staging_bucket"],
    )

    # Compile pipeline
    compiled_pipeline = compile_pipeline()

    # Unique run ID
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_id    = f"purchase-prediction-{timestamp}"

    logger.info(f"Submitting pipeline: {job_id}")

    # Pipeline parameters
    pipeline_params = {
        "project_id":             gcp["project_id"],
        "region":                 gcp["region"],
        "bq_dataset":             config["bigquery"]["dataset"],
        "bq_table":               config["bigquery"]["table"],
        "staging_bucket":         gcp["staging_bucket"],
        "query_limit":            config["bigquery"]["query_limit"],
        "n_estimators":           hp.get("n_estimators", 300),
        "max_depth":              hp.get("max_depth", 6),
        "learning_rate":          hp.get("learning_rate", 0.05),
        "subsample":              hp.get("subsample", 0.8),
        "accuracy_threshold":     ev["accuracy_threshold"],
        "f1_threshold":           ev["f1_threshold"],
        "model_display_name":     tr["model_display_name"],
        "endpoint_display_name":  f"{tr['model_display_name']}-endpoint",
        "machine_type":           srv["machine_type"],
    }

    # Submit job
    job = aiplatform.PipelineJob(
        display_name=job_id,
        template_path=compiled_pipeline,
        pipeline_root=pipe["pipeline_root"],
        parameter_values=pipeline_params,
        enable_caching=False,
    )

    job.submit(
        service_account=(
            f"vertex-mlops-sa@{gcp['project_id']}.iam.gserviceaccount.com"
        )
    )

    logger.info(f"Pipeline submitted! Job ID: {job_id}")
    logger.info(
        f"Monitor at: https://console.cloud.google.com/vertex-ai/pipelines"
        f"?project={gcp['project_id']}"
    )
    logger.info(f"Resource name: {job.resource_name}")


def main():
    parser = argparse.ArgumentParser(description="Run Vertex AI ML Pipeline")
    parser.add_argument(
        "--config", default="configs/pipeline_config.yaml",
        help="Path to pipeline config YAML",
    )
    parser.add_argument(
        "--wait", action="store_true",
        help="Wait for pipeline to complete",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    logger.info(
        f"Running pipeline for project: {config['gcp']['project_id']}"
    )
    run_pipeline(config)


if __name__ == "__main__":
    main()
