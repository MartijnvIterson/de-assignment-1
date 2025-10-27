## UI app for short-lived deployment

We've added a small Flask-based UI in `app/` that downloads a registered model from GCS and offers a CSV file upload to run batch predictions. The UI assets are split across `templates/index.html`, `static/styles.css` and `static/app.js` for clarity.
Dockerfile: `Dockerfile.app` builds a small image that can be pushed to Container Registry (gcr.io) or Artifact Registry and deployed to Cloud Run.

Recommended (PowerShell) commands to build/push (replace PROJECT and REGION):

```powershell
# build
docker build -f Dockerfile.app -t gcr.io/$env:PROJECT/fraud-ui:latest .

# push
docker push gcr.io/$env:PROJECT/fraud-ui:latest

Deployment is invoked from the pipeline using `gcloud run deploy`. The pipeline will only deploy when the new model is considered better and registration succeeded. After deployment the pipeline triggers a cleanup component that waits ~30 minutes (1800s) and deletes the Cloud Run service to limit costs.

Cost considerations:
- Keep the Cloud Run service minimal: single concurrent instance, only 30 minutes runtime.
- Use the smallest Cloud Run CPU/memory if needed via `--cpu` and `--memory` flags (not currently set in the pipeline to keep simplicity).
- Aim to keep overall run cost under $0.50 by using short runtime and Cloud Run free tier (varies by region/account).

Automated scheduling (done)
---------------------------
The pipeline now includes a `deploy_and_schedule_teardown` component that:
- deploys the UI image to Cloud Run, and
- schedules a Cloud Tasks task to call the teardown Cloud Function after a delay (default 1800s = 30 min).

Pipeline parameters added:
- `teardown_function_url` : URL of the HTTP Cloud Function that deletes the Cloud Run service.
- `tasks_queue` : name of the Cloud Tasks queue to use (must already exist).
- `tasks_location` : region of the queue (default 'us-central1').

This makes the whole flow automated: pipeline deploys the UI and immediately schedules the asynchronous teardown. No manual runs of scripts in `tools/` are needed unless you prefer to manage scheduling externally.

Teardown orchestration (non-blocking)
------------------------------------
To avoid keeping the pipeline run open while waiting 30 minutes, we provide a small Cloud Function + Cloud Tasks pattern:

- `tools/teardown_function/main.py` : a simple Cloud Function that deletes a Cloud Run service when invoked (expects JSON payload with project_id and service_name).
- `tools/schedule_teardown.py` : helper that creates a Cloud Tasks task scheduled ~30 minutes in the future to call the Cloud Function URL.

How to use from the pipeline:
1. Deploy the Cloud Function `teardown_cloud_run` (make it HTTP-triggered and allow invocation by Cloud Tasks service account).
2. Ensure you have a Cloud Tasks queue in the same region (or adapt location).
3. After `deploy_to_cloud_run` completes, call (from a small step or as part of your deploy component) the helper to create a scheduled task that will post to the Cloud Function URL after 1800s. This keeps pipeline short and still ensures automatic deletion.

If you want, I can wire the image-deploy component so it calls `create_teardown_task` automatically (requires adding the Cloud Tasks client to the deploy container and specifying queue/url details). Say the word and I implement that extra step.
# Credit Card Fraud Detection MLOps Pipeline

## Project Overview

This project implements a complete Machine Learning Operations (MLOps) pipeline for credit card fraud detection. The pipeline uses Google Cloud Platform (GCP) services to automatically train, evaluate, and deploy XGBoost models for detecting fraudulent transactions.

### What this project does exactly:

1. **Automated ML Model Training**: Trains XGBoost models on credit card transaction data to detect fraud
2. **Kubeflow Pipeline**: Uses Google Cloud Vertex AI Pipelines for structured ML workflows
3. **Component-based Architecture**: 5 reusable ML components for data loading, preprocessing, training, evaluation, and deployment
4. **Docker Container Deployment**: All components run in Docker containers for consistency
5. **Cloud Build Integration**: Automated builds and deployments via Google Cloud Build
6. **Artifact Storage**: Stores models, data, and results in Google Cloud Storage buckets

## Current Project Status

⚠️ **Note**: The PROJECT_ID `data-engineering-jads-2025` is already configured and operational in the Google Cloud environment. This PROJECT_ID is hardcoded in the following locations:

### Hardcoded PROJECT_ID Locations:
- **visual_pipeline.py** (line 299): `project_id: str = "data-engineering-jads-2025"`
- **cloudbuild/cloudbuild.simple.yaml** (line 81): `_PROJECT_ID: 'data-engineering-jads-2025'`
- **cloudbuild/cloudbuild.simple.yaml** (line 92): Service account reference
- **.github/workflows/simple-mlops.yml** (lines 34, 38): Cloud Build project parameter
- **compile_and_run_pipeline.py** (line 48): Default PROJECT_ID fallback
- **pipeline_summary.py** (line 18): Default PROJECT_ID fallback
- **train_local.py** (line 156): Monitoring URL

## Pipeline Architecture

The ML pipeline consists of 5 main components:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Data Loading   │───▶│ Preprocessing   │───▶│   Training      │
│                 │    │                 │    │                 │
│ • Loads dataset │    │ • Train/test    │    │ • XGBoost       │
│   from GCS      │    │   split         │    │   model         │
│ • Validation    │    │ • Feature eng.  │    │ • Hyperparams   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐             │
│   Deployment    │◀───│   Evaluation    │◀────────────┘
│                 │    │                 │
│ • Model deploy  │    │ • ROC-AUC       │
│ • Serving       │    │ • PR-AUC        │
│ • Monitoring    │    │ • Confusion     │
└─────────────────┘    └─────────────────┘
```

## Used Google Cloud Services

### Active Services:
- **Vertex AI Pipelines**: For ML workflow orchestration
- **Cloud Build**: For automated builds and deployments  
- **Cloud Storage**: For data and artifact storage (4 buckets)
- **Container Registry/Docker Hub**: For component images
- **IAM**: For service account management

### Bucket Configuration:
The project uses 4 Google Cloud Storage buckets:
- `data-engineering-jads-2025-data`: Dataset storage (creditcard.csv)
- `data-engineering-jads-2025-pipeline-specs`: Pipeline definitions
- `data-engineering-jads-2025-pipeline-artifacts`: Intermediate results
- `data-engineering-jads-2025-models`: Trained models

## Security & Secrets Configuration

### GitHub Secrets (Required):
For automated deployments, the following secrets must be set in GitHub:

```bash
# Google Cloud Platform Secrets
GCP_PROJECT_ID=data-engineering-jads-2025
GCP_SA_KEY=[Complete JSON content of service account key file]
GCP_REGION=us-central1

# Docker Hub Secrets (for container registry)
DOCKERHUB_USERNAME=[your-dockerhub-username]
DOCKERHUB_TOKEN=dckr_pat_[your-docker-hub-access-token]
```

### Service Account Configuration:
The pipeline uses the service account: `github-actions@data-engineering-jads-2025.iam.gserviceaccount.com`

**Required Roles:**
- Vertex AI User (roles/aiplatform.user)
- Cloud Build Editor (roles/cloudbuild.builds.editor)
- Storage Admin (roles/storage.admin)
- Secret Manager Secret Accessor (roles/secretmanager.secretAccessor)
- Service Account User (roles/iam.serviceAccountUser)

### Docker Hub Setup:
For the Docker Hub token:
1. Go to Docker Hub → Account Settings → Security
2. Click "New Access Token"
3. Name: `mlops-pipeline`
4. Permissions: Read, Write
5. Copy the generated token (starts with `dckr_pat_`)

## File Structure

```
creditcard-fraud-mlops/
├── visual_pipeline.py              # Main pipeline definition with 5 components
├── compile_and_run_pipeline.py     # Pipeline compiler and Vertex AI submitter
├── train_local.py                  # Local training script for testing
├── pipeline_summary.py             # Pipeline status and monitoring
├── requirements.txt                # Python dependencies
├── cloudbuild/
│   └── cloudbuild.simple.yaml     # Cloud Build configuration
├── docker/
│   └── Dockerfile.components       # Docker image for ML components
└── .github/workflows/
    └── simple-mlops.yml           # GitHub Actions workflow
```

## Pipeline Execution

### Automatic Trigger:
- **Push to main branch**: Automatically starts Cloud Build
- **GitHub Actions**: Executes builds and triggers Vertex AI pipeline
- **Vertex AI**: Runs the ML pipeline with all 5 components

### Monitoring:
1. **GitHub Actions**: https://github.com/[repository]/actions
2. **Cloud Build**: https://console.cloud.google.com/cloud-build/builds?project=data-engineering-jads-2025  
3. **Vertex AI Pipelines**: https://console.cloud.google.com/vertex-ai/pipelines?project=data-engineering-jads-2025

## Requirements for Execution

### Python Dependencies:
```
google-cloud-aiplatform==1.36.0
google-cloud-storage==2.10.0
pandas==2.0.3
scikit-learn==1.3.0
xgboost==1.7.6
numpy==1.24.3
kfp==2.4.0
```

### Dataset:
- **Source**: Credit Card Fraud Dataset from Kaggle
- **Location**: `gs://data-engineering-jads-2025-data/creditcard.csv`
- **Format**: CSV with transaction features and 'Class' label (0=normal, 1=fraud)

## Manual Pipeline Execution

For local development and testing:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Compile pipeline
python compile_and_run_pipeline.py

# 3. Run local training (for testing)
python train_local.py

# 4. Check pipeline status
python pipeline_summary.py
```

## Cost Monitoring

**Estimated monthly costs:**
- Cloud Storage: ~€5-10 (datasets, artifacts)
- Vertex AI Compute: ~€15-25 (pipeline executions)
- Cloud Build: ~€5 (automated builds)
- **Total**: ~€25-40/month

## Troubleshooting

### Common Issues:

1. **Pipeline fails with permission errors**
   - Check service account roles
   - Verify that all APIs are enabled

2. **Docker image push fails**
   - Check DOCKERHUB_TOKEN in GitHub secrets
   - Verify Docker Hub repository access

3. **Dataset not found error**
   - Check if creditcard.csv is in the correct bucket
   - Verify bucket naming: `data-engineering-jads-2025-data`

4. **Build failures in Cloud Build**
   - Check Cloud Build service account permissions
   - Verify that all GCP APIs are enabled

### Monitoring Tools:
- **Cloud Logging**: For detailed error logs
- **Vertex AI Console**: For pipeline execution monitoring  
- **GitHub Actions Logs**: For build and deployment logs

## Next Steps

For further development of the pipeline:

1. **Model Performance Monitoring**: Implement model drift detection
2. **Data Quality Checks**: Add data validation components
3. **A/B Testing**: Implement model comparison framework
4. **Real-time Serving**: Setup model serving endpoints
5. **Alerting**: Configure monitoring alerts for pipeline failures

---

*This README describes the current status of the MLOps project with focus on the operational aspects and configuration of the existing Google Cloud environment.*