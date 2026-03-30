#!/usr/bin/env bash
# =============================================================
# setup_gcp.sh — One-time GCP resource setup
# Run this ONCE before triggering the pipeline.
# =============================================================

set -euo pipefail

# ── Load values from config (or set manually below) ──────────────────────────
PROJECT_ID="${GCP_PROJECT_ID:-your-gcp-project-id}"   # ← set env var or edit
REGION="${GCP_REGION:-us-central1}"
BUCKET_NAME="${GCP_BUCKET:-${PROJECT_ID}-mlops-vertexai}"
REPO_NAME="mlops-repo"

echo "=========================================="
echo "  MLOps GCP Setup"
echo "  Project : $PROJECT_ID"
echo "  Region  : $REGION"
echo "  Bucket  : $BUCKET_NAME"
echo "=========================================="

# ── Authenticate & set project ────────────────────────────────────────────────
gcloud config set project "$PROJECT_ID"

# ── Enable required APIs ──────────────────────────────────────────────────────
echo "Enabling GCP APIs ..."
gcloud services enable \
    aiplatform.googleapis.com \
    bigquery.googleapis.com \
    bigquerystorage.googleapis.com \
    storage.googleapis.com \
    artifactregistry.googleapis.com \
    cloudbuild.googleapis.com \
    iam.googleapis.com \
    --project="$PROJECT_ID"

echo "APIs enabled ✓"

# ── Create GCS Bucket ─────────────────────────────────────────────────────────
if gsutil ls -b "gs://$BUCKET_NAME" &>/dev/null; then
    echo "Bucket gs://$BUCKET_NAME already exists."
else
    gsutil mb -l "$REGION" "gs://$BUCKET_NAME"
    echo "Bucket gs://$BUCKET_NAME created ✓"
fi

# ── Create Artifact Registry (Docker repo) ────────────────────────────────────
if gcloud artifacts repositories describe "$REPO_NAME" \
    --location="$REGION" --project="$PROJECT_ID" &>/dev/null; then
    echo "Artifact Registry repo '$REPO_NAME' already exists."
else
    gcloud artifacts repositories create "$REPO_NAME" \
        --repository-format=docker \
        --location="$REGION" \
        --description="MLOps Docker images" \
        --project="$PROJECT_ID"
    echo "Artifact Registry '$REPO_NAME' created ✓"
fi

# ── Create BigQuery Dataset ───────────────────────────────────────────────────
BQ_DATASET="ecommerce_data"
if bq ls --datasets --project_id="$PROJECT_ID" | grep -q "$BQ_DATASET"; then
    echo "BigQuery dataset '$BQ_DATASET' already exists."
else
    bq mk --dataset --location="$REGION" "${PROJECT_ID}:${BQ_DATASET}"
    echo "BigQuery dataset '$BQ_DATASET' created ✓"
fi

# ── Create service account for Vertex AI ─────────────────────────────────────
SA_NAME="vertex-mlops-sa"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

if gcloud iam service-accounts describe "$SA_EMAIL" --project="$PROJECT_ID" &>/dev/null; then
    echo "Service account '$SA_EMAIL' already exists."
else
    gcloud iam service-accounts create "$SA_NAME" \
        --display-name="Vertex AI MLOps Service Account" \
        --project="$PROJECT_ID"
    echo "Service account created ✓"
fi

# Grant required roles
for ROLE in \
    roles/aiplatform.user \
    roles/bigquery.dataViewer \
    roles/bigquery.jobUser \
    roles/storage.objectAdmin \
    roles/artifactregistry.writer; do
    gcloud projects add-iam-policy-binding "$PROJECT_ID" \
        --member="serviceAccount:$SA_EMAIL" \
        --role="$ROLE" --quiet
done
echo "IAM roles granted ✓"

# ── Upload sample data to BigQuery ────────────────────────────────────────────
echo "Uploading sample data to BigQuery ..."
if [ -f "data/sample_data.csv" ]; then
    bq load \
        --autodetect \
        --source_format=CSV \
        --skip_leading_rows=1 \
        "${PROJECT_ID}:${BQ_DATASET}.customer_transactions" \
        "data/sample_data.csv"
    echo "Sample data loaded into BigQuery ✓"
else
    echo "Warning: data/sample_data.csv not found. Generate it first with:"
    echo "  python scripts/generate_sample_data.py"
fi

# ── Update pipeline_config.yaml ───────────────────────────────────────────────
echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Update configs/pipeline_config.yaml with:"
echo "     gcp.project_id: $PROJECT_ID"
echo "     gcp.staging_bucket: gs://$BUCKET_NAME/mlops-staging"
echo ""
echo "  2. Run the pipeline:"
echo "     python scripts/run_pipeline.py --config configs/pipeline_config.yaml"
echo ""
echo "  Service Account: $SA_EMAIL"
echo "  Bucket         : gs://$BUCKET_NAME"
