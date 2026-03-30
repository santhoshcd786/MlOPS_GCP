# 🛒 Customer Purchase Prediction — End-to-End MLOps on GCP Vertex AI

> Predict the **highest-priced item** a customer is most likely to buy next, using Google Cloud Platform and Vertex AI Pipelines.

---

## 🏗️ Architecture Overview

```
Raw Data (BigQuery)
      │
      ▼
┌─────────────────────────────────────────────┐
│           Vertex AI Pipeline                │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐ │
│  │ Ingest & │→ │  Feature  │→ │  Train   │ │
│  │ Validate │  │ Engineer  │  │  Model   │ │
│  └──────────┘  └───────────┘  └──────────┘ │
│                                ┌──────────┐ │
│                                │ Evaluate │ │
│                                └──────────┘ │
│                                ┌──────────┐ │
│                                │  Deploy  │ │
│                                └──────────┘ │
└─────────────────────────────────────────────┘
      │
      ▼
Vertex AI Endpoint (REST API)
      │
      ▼
Monitoring (Vertex AI Model Monitoring)
```

---

## 📁 Project Structure

```
mlops-vertexai-project/
├── configs/                    # Environment & pipeline configs
│   ├── pipeline_config.yaml
│   └── model_config.yaml
├── data/                       # Sample & schema files
│   └── sample_data.csv
├── src/
│   ├── features/               # Feature engineering
│   │   └── feature_engineering.py
│   ├── models/                 # Model training & evaluation
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── pipelines/              # Vertex AI Kubeflow Pipelines
│   │   └── training_pipeline.py
│   └── serving/                # Prediction serving logic
│       └── predictor.py
├── scripts/                    # Helper scripts
│   ├── setup_gcp.sh
│   ├── run_pipeline.py
│   └── deploy_model.py
├── notebooks/                  # EDA & experimentation
│   └── 01_eda_and_baseline.ipynb
├── tests/                      # Unit tests
│   ├── test_features.py
│   └── test_model.py
├── .github/workflows/          # CI/CD
│   └── mlops_cicd.yaml
├── docs/
│   └── architecture.md
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🎯 Problem Statement

**Goal:** Given a customer's purchase history and behavioral signals, predict:
1. The **category** of item the customer will buy next
2. The **price tier** (highest-priced item they're likely to purchase)

**Business Value:**
- Personalized upselling recommendations
- Targeted marketing campaigns
- Inventory forecasting

---

## 🔧 Tech Stack

| Layer | Technology |
|-------|-----------|
| Cloud | Google Cloud Platform (GCP) |
| ML Platform | Vertex AI |
| Pipelines | Vertex AI Pipelines (KFP v2) |
| Feature Store | Vertex AI Feature Store |
| Model Registry | Vertex AI Model Registry |
| Serving | Vertex AI Endpoints |
| Monitoring | Vertex AI Model Monitoring |
| Data Warehouse | BigQuery |
| Storage | Google Cloud Storage (GCS) |
| CI/CD | GitHub Actions |
| Containerization | Docker + Artifact Registry |
| Language | Python 3.10+ |
| ML Framework | XGBoost + Scikit-learn |

---

## 🚀 Quick Start

### 1. Prerequisites

```bash
# Install Google Cloud SDK
# https://cloud.google.com/sdk/docs/install

gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

### 2. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/mlops-vertexai-project.git
cd mlops-vertexai-project

pip install -r requirements.txt
```

### 3. Configure GCP

```bash
# Set up GCP resources (run once)
bash scripts/setup_gcp.sh
```

Update `configs/pipeline_config.yaml` with your GCP project details.

### 4. Run the ML Pipeline

```bash
python scripts/run_pipeline.py --config configs/pipeline_config.yaml
```

### 5. Deploy the Model

```bash
python scripts/deploy_model.py --model-display-name purchase-predictor-v1
```

### 6. Make Predictions

```bash
curl -X POST \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  -d @data/sample_request.json \
  https://REGION-aiplatform.googleapis.com/v1/projects/PROJECT_ID/locations/REGION/endpoints/ENDPOINT_ID:predict
```

---

## 📊 Model Details

### Features Used

| Feature | Description | Type |
|---------|-------------|------|
| `customer_age` | Age of customer | Numeric |
| `avg_purchase_value` | Average historical spend | Numeric |
| `purchase_frequency` | Purchases per month | Numeric |
| `days_since_last_purchase` | Recency signal | Numeric |
| `top_category` | Most bought category | Categorical |
| `browsing_time_mins` | Avg session duration | Numeric |
| `cart_abandonment_rate` | Behavioral signal | Numeric |
| `loyalty_tier` | Bronze/Silver/Gold | Categorical |
| `season` | Quarter of year | Categorical |
| `device_type` | Mobile/Desktop/Tablet | Categorical |

### Target Variable
`highest_price_item_category` — The category of the most expensive item the customer is predicted to purchase in the next 30 days.

### Model Performance (on holdout set)

| Metric | Value |
|--------|-------|
| Accuracy | ~84% |
| F1 Score (weighted) | ~0.83 |
| ROC-AUC (macro) | ~0.91 |

---

## 🔄 MLOps Pipeline Stages

```
1. DATA INGESTION      → Pull from BigQuery, validate schema
2. FEATURE ENGINEERING → Create customer behavioral features  
3. DATA SPLIT          → Train / Validation / Test (70/15/15)
4. MODEL TRAINING      → XGBoost classifier with hyperparameter tuning
5. MODEL EVALUATION    → Accuracy, F1, Confusion matrix
6. CONDITIONAL GATE    → Only deploy if accuracy > threshold (80%)
7. MODEL REGISTRY      → Register model in Vertex AI Model Registry
8. DEPLOYMENT          → Deploy to Vertex AI Endpoint
9. MONITORING          → Track data drift & prediction drift
```

---

## 🔁 CI/CD with GitHub Actions

Every push to `main`:
1. Runs unit tests (`pytest`)
2. Builds & pushes Docker image to Artifact Registry
3. Triggers Vertex AI Pipeline run
4. Auto-deploys if evaluation passes

---

## 📈 Monitoring

- **Data Drift Detection**: Monitors input feature distributions
- **Prediction Drift**: Alerts if output distribution shifts
- **Performance Degradation**: Scheduled retraining trigger

---

## 🤝 Contributing

1. Fork the repo
2. Create feature branch (`git checkout -b feature/my-feature`)
3. Commit changes (`git commit -m 'Add feature'`)
4. Push and open a Pull Request

---
