# Architecture Documentation

## System Overview

This MLOps project implements a production-grade machine learning system on **Google Cloud Platform (GCP)** using **Vertex AI** for end-to-end pipeline orchestration.

## Data Flow

```
[Raw E-commerce Data]
         │
         ▼
[BigQuery — Data Warehouse]
  customer_transactions table
         │
         ▼
[Vertex AI Pipeline — KFP v2]
  ┌──────────────────────────────────────────────────────┐
  │                                                      │
  │  Step 1: Data Ingestion                              │
  │  └─ Pull from BigQuery via SQL query                 │
  │  └─ Schema validation                                │
  │                                                      │
  │  Step 2: Feature Engineering                         │
  │  └─ RFM (Recency/Frequency/Monetary) features        │
  │  └─ Behavioral features                              │
  │  └─ Categorical encoding                             │
  │  └─ Target variable creation                         │
  │                                                      │
  │  Step 3: Model Training                              │
  │  └─ XGBoost multi-class classifier                   │
  │  └─ Early stopping on validation set                 │
  │  └─ K-fold cross-validation                          │
  │                                                      │
  │  Step 4: Model Evaluation + Gate                     │
  │  └─ Accuracy, F1, ROC-AUC on holdout                 │
  │  └─ Conditional: deploy only if acc > 80%            │
  │                                                      │
  │  Step 5: Model Registry                              │
  │  └─ Upload to Vertex AI Model Registry               │
  │  └─ Versioned model artifacts on GCS                 │
  │                                                      │
  │  Step 6: Deployment                                  │
  │  └─ Vertex AI Endpoint (auto-scaled)                 │
  │  └─ REST API for real-time predictions               │
  │                                                      │
  └──────────────────────────────────────────────────────┘
         │
         ▼
[Vertex AI Endpoint — REST API]
  POST /predict
  Returns: predicted category + confidence + price tier
         │
         ▼
[Vertex AI Model Monitoring]
  └─ Data drift detection
  └─ Prediction drift alerts
  └─ Scheduled retraining triggers
```

## Key Design Decisions

### Why XGBoost?
- Strong baseline performance on tabular e-commerce data
- Fast training and inference
- Native support in Vertex AI serving containers
- Built-in feature importance

### Why KFP v2 Pipelines?
- Native integration with Vertex AI
- Each step is a containerized component (reproducible)
- Automatic caching of intermediate results
- Visual pipeline graph in GCP Console

### Deployment Gate
The pipeline includes a conditional check before deploying:
- Accuracy must exceed **80%**
- F1 (weighted) must exceed **78%**
- If model fails gate → pipeline stops, no bad model reaches production

### Scaling
- Vertex AI Endpoints auto-scale from min 1 → max 3 replicas
- Controlled by request load

## Cost Estimates (approximate)

| Resource | Estimated Monthly Cost |
|----------|----------------------|
| Vertex AI Training (n1-standard-4 × 2h/week) | ~$15 |
| Vertex AI Endpoint (n1-standard-2 × 1 replica) | ~$55 |
| BigQuery (100K rows/week queries) | < $1 |
| GCS Storage (model artifacts) | < $1 |
| **Total** | **~$71/month** |

## Security

- Service account with least-privilege IAM roles
- No credentials stored in code (env vars / Secret Manager)
- Docker images scanned for CVEs via Trivy in CI

## Monitoring & Alerting

- **Data Drift**: Monitors feature distributions against training baseline
- **Prediction Drift**: Alerts if output distribution shifts > threshold
- **Email Alerts**: Configured in `pipeline_config.yaml`
- **Retraining**: Triggered automatically via Cloud Scheduler (weekly)
