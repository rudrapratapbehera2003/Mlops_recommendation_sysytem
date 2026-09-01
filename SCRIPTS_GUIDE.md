# Project Scripts Documentation

## Overview

This project includes several automation scripts for development, testing, and deployment. Here's what each script does and how to use it.

---

## 1. `scripts/validate_deployment.py`

### Purpose
Comprehensive pre-deployment validation script that checks the entire project is ready for production deployment.

### What It Does
- Validates all required environment variables are set
- Loads and validates configuration from YAML
- Verifies inference module imports correctly
- Validates Docker Compose configuration
- Runs entire test suite
- Checks code quality (formatting, imports)
- Tests API endpoints if running locally
- Generates final deployment readiness report

### When to Use
- Before deploying to production
- After making code changes
- In CI/CD pipeline (GitHub Actions)
- To verify project integrity

### How to Run

**Option 1: With Docker (recommended)**
```powershell
docker compose exec api python scripts/validate_deployment.py
```

**Option 2: Locally**
```powershell
# Activate virtual environment first
.\venv\Scripts\Activate.ps1

# Run validation
python scripts/validate_deployment.py
```

### Expected Output
```
Starting end-to-end deployment validation...

✓ Environment Variables
✓ Config Loading
✓ Inference Module
✓ Docker Compose
✓ Test Suite
✓ Code Quality
✓ API Endpoints
============================================================
DEPLOYMENT VALIDATION REPORT
============================================================
✓ Environment Variables
✓ Config Loading
✓ Inference Module
✓ Docker Compose
✓ Test Suite
✓ Code Quality
✓ API Endpoints
============================================================
Result: 7/7 checks passed
Status: ✓ READY FOR DEPLOYMENT
```

### Exit Codes
- `0` — All checks passed, ready to deploy
- `1` — Some non-critical checks failed, proceed with caution
- `2` — Critical checks failed, do not deploy

---

## 2. `scripts/deploy_checklist.sh`

### Purpose
Step-by-step deployment checklist that verifies everything is ready before production deployment.

### What It Does
- Runs `validate_deployment.py` checks
- Builds Docker image for production
- Validates docker-compose configuration
- Verifies model artifact exists
- Tests MLflow connectivity
- Provides final go/no-go decision

### When to Use
- Final check before deploying to production
- On deployment server
- As part of release procedure

### How to Run

**On Linux/Mac:**
```bash
bash scripts/deploy_checklist.sh
```

**On Windows (PowerShell):**
```powershell
# Convert bash script to PowerShell or use WSL
wsl bash scripts/deploy_checklist.sh

# Or run individual checks manually
python scripts/validate_deployment.py
docker compose config --quiet
docker build -t course-recommendation-api:latest .
```

### Expected Output
```
==========================================
Production Deployment Checklist
==========================================

1. Environment validation...
✓ Environment variables validated
✓ Config file loaded successfully
✓ Inference module imports successfully
✓ Docker Compose configuration is valid
✓ All tests passed
✓ Code Quality passed
✓ API Endpoints reachable

2. Building Docker image...
[+] Building 45.2s (15/15)
...
Successfully tagged course-recommendation-api:latest

3. Validating docker-compose for production...

4. Checking model artifact...
✓ Model artifact found

5. Verifying MLflow connectivity...
✓ MLflow server is accessible

6. Final checks...
✓ All pre-deployment checks passed

Ready to deploy. Run: docker-compose up -d
```

---

## 3. `src/pipelines/training_pipeline.py`

### Purpose
End-to-end machine learning pipeline that orchestrates data ingestion, validation, feature engineering, model training, and model registry operations.

### What It Does

1. **Data Ingestion**
   - Reads raw data from Excel file
   - Handles missing values (mean/median imputation)
   - Saves cleaned data to CSV

2. **Data Validation**
   - Validates data types
   - Checks for null values
   - Validates value ranges
   - Generates validation report

3. **Feature Engineering**
   - Clips outliers in numerical features
   - Creates engineered features
   - Saves feature-engineered dataset

4. **Model Training**
   - Trains Collaborative Filtering (SVD) model
   - Trains Content-Based Filtering (RandomForest) model
   - Evaluates both models
   - Selects best model
   - Registers model in MLflow model registry
   - Promotes model to Staging stage

5. **Artifact Management**
   - Saves final model bundle (both CF and CBF models)
   - Logs artifacts to MLflow
   - Tracks model metadata

### When to Use
- Initial training on raw data
- Retraining with new data
- Scheduled training (via Airflow)
- Model updates and version management

### How to Run

**Option 1: Local (Standalone)**
```powershell
# Make sure you have activated the virtual environment
.\venv\Scripts\Activate.ps1

# Run the training pipeline
python src\pipelines\training_pipeline.py
```

**Option 2: Docker Compose (Recommended)**
```powershell
# Start services first (if not already running)
docker compose up -d

# Run training inside API container
docker compose exec api python src/pipelines/training_pipeline.py
```

**Option 3: Airflow DAG**
```powershell
# Open Airflow UI
Start-Process "http://localhost:8080"

# Trigger the DAG manually or wait for scheduled run (daily at 2 AM)
```

### Expected Output

```
[2026-08-31 16:36:06] INFO [__main__:25] - Starting end-to-end training pipeline.
[2026-08-31 16:36:06] INFO [src.data.ingest:77] - Starting data ingestion workflow...
[2026-08-31 16:36:06] INFO [src.data.ingest:37] - Successfully located raw data file at: data/raw/online_course_recommendation_v2.xlsx
[2026-08-31 16:36:14] INFO [src.data.ingest:72] - Baseline data successfully saved to: data/processed/cleaned_recommend_data.csv
[2026-08-31 16:36:14] INFO [src.data.ingest:81] - Data ingestion workflow completed successfully.

[2026-08-31 16:36:14] INFO [src.data.validate:36] - Starting comprehensive data validation checks...
[2026-08-31 16:36:14] INFO [src.data.validate:85] - All validation constraints passed successfully.
[2026-08-31 16:36:14] INFO [src.data.validate:109] - Validation report saved to: data/validation_reports\latest_validation_report.txt

[2026-08-31 16:36:14] INFO [src.features.build_features:66] - Starting feature enginnering workflow....
[2026-08-31 16:36:14] INFO [src.features.build_features:30] - Successfully located raw data file at: data/processed/cleaned_recommend_data.csv
[2026-08-31 16:36:14] INFO [src.features.build_features:37] - Starting the clipping of outliers in the numerical variables......
[2026-08-31 16:36:14] INFO [src.features.build_features:72] - Feature enginnering successfull. Matrix saved to: data/processed/engineered_recommend_data.csv

[2026-08-31 16:36:15] INFO [src.models.train_model:191] - Starting model selection and training process...
[2026-08-31 16:36:53] INFO [src.models.train_model:53] - Starting the implementation of colaborative filtering model.......
[2026-08-31 16:37:01] INFO [src.models.train_model:70] - Successfully trained colaborative filtering model......
[2026-08-31 16:37:02] INFO [src.models.train_model:102] - Evaluating  CF Model.
[2026-08-31 16:37:02] INFO [src.models.train_model:115] - CF Model Metrices values --> RMSE:1.234, MAE:0.987

[2026-08-31 16:37:03] INFO [src.models.train_model:122] - Starting the implementation of contentet based filtering model using RandomForestRegressor model.
[2026-08-31 16:37:45] INFO [src.models.train_model:128] - Successfully trained the CBF Model using RandomForestRegressor.
[2026-08-31 16:37:45] INFO [src.models.train_model:131] - Evaluating CBF Model
[2026-08-31 16:37:45] INFO [src.models.train_model:134] - CBF RandomForestRegressor Model Metrices --> RMSE:1.123, MAE:0.876

[2026-08-31 16:37:45] INFO [src.models.train_model:201] - Best model selected: cbf
[2026-08-31 16:37:50] INFO [src.models.train_model:273] - Final model saved to: models/recommender_model.joblib
[2026-08-31 16:37:50] INFO [src.models.train_model:281] - Registered model version: {'name': 'course_recommendation', 'version': '1', 'stage': 'Staging', 'run_id': 'abc123...', 'source': 'runs:/abc123.../content_based_random_forest'}

[2026-08-31 16:37:50] INFO [__main__:45] - Training pipeline completed successfully.
```

### Output Files Created

- `data/processed/cleaned_recommend_data.csv` — Cleaned data after handling missing values
- `data/processed/engineered_recommend_data.csv` — Feature-engineered data
- `data/validation_reports/latest_validation_report.txt` — Validation report
- `models/recommender_model.joblib` — Final model bundle with both CF and CBF models
- MLflow artifacts and metadata stored in `mlartifacts/`

### Troubleshooting

**Error: "Enginnered file is missing"**
- Feature engineering failed; check logs above
- Verify cleaned data file exists at `data/processed/cleaned_recommend_data.csv`

**Error: "MLflow tracking URI unreachable"**
- MLflow server not running; start it: `docker compose up mlflow -d`
- Check `.env` file for correct MLFLOW_TRACKING_URI

**Error: "Model training timed out"**
- Large dataset or slow machine; increase timeout
- Check available memory: `Get-Process | Select-Object -First 5 | Format-Table Name, WorkingSet`

---

## 4. `scripts/validate_deployment.py`

### Purpose (Already Described Above)
See Section 1 for full details.

---

## 5. `scripts/deploy_checklist.sh`

### Purpose (Already Described Above)
See Section 2 for full details.


## 3. Source Code Pipeline Scripts

### `src/pipelines/training_pipeline.py`

**Purpose**: Complete end-to-end training pipeline orchestration

**What It Does**:
1. Ingests raw data from Excel file
2. Validates data quality
3. Engineers features
4. Trains both models:
   - Collaborative Filtering (TruncatedSVD)
   - Content-Based Filtering (RandomForestRegressor)
5. Evaluates both models
6. Registers best model to MLflow
7. Promotes model to Staging stage
8. Saves model bundle with preprocessing

**How to Run**:

```powershell
# With Docker
docker compose exec api python src/pipelines/training_pipeline.py

# Locally
python src/pipelines/training_pipeline.py
```

**Expected Output**:
```
2026-08-31 15:30:00 - INFO - Starting end-to-end training pipeline.
2026-08-31 15:30:05 - INFO - Data ingestion completed
2026-08-31 15:30:10 - INFO - Data validation passed
2026-08-31 15:30:15 - INFO - Feature engineering completed
2026-08-31 15:30:45 - INFO - Starting model selection and training process...
2026-08-31 15:31:00 - INFO - Best model selected: content_based_random_forest
2026-08-31 15:31:30 - INFO - Final model saved to: models/recommender_model.joblib
2026-08-31 15:31:35 - INFO - Registered model version: {...}
2026-08-31 15:31:35 - INFO - Training pipeline completed successfully.
```

### `src/models/train_model.py`

**Purpose**: Model training and registry integration

**Key Methods**:
- `implementing_cf_model()` — Trains Collaborative Filtering model
- `implementing_randomforest_cbf()` — Trains Content-Based model
- `evaluate_cf_model()` — Evaluates CF model performance
- `evaluate_cbf_model()` — Evaluates CBF model performance
- `_register_model_version()` — Registers model to MLflow registry
- `compairing_to_find_best_model()` — Main orchestration method

**How to Run**:

```powershell
# Run training script directly
python src/models/train_model.py

# Or via pipeline
python src/pipelines/training_pipeline.py
```

---

## 4. Inference Script

### `src/inference/predict.py`

**Purpose**: Load trained models and generate recommendations

**Key Functions**:
- `load_model()` — Loads model bundle from joblib
- `_recommend_for_cbf()` — Content-Based recommendations
- `_recommend_for_cf()` — Collaborative Filtering recommendations
- `predict_recommendations()` — Main inference function

**Usage Example**:

```python
from src.inference.predict import predict_recommendations

# Content-Based recommendation
result = predict_recommendations(
    user_id=1,
    top_n=5,
    model_type="content_based_random_forest",
    candidate_features=[
        {"course_id": 101, "course_duration_hours": 20, "rating": 4.5}
    ]
)

# Collaborative Filtering recommendation
result = predict_recommendations(
    user_id=5,
    top_n=10,
    model_type="collaborative_filtering",
    candidate_course_ids=[101, 102, 103, 104]
)

print(result)
```

---

## 5. Airflow DAG

### `airflow/dags/recommendation_training_dag.py`

**Purpose**: Scheduled automated model training via Apache Airflow

**What It Does**:
- Runs training pipeline on a schedule (daily at 2 AM)
- Catches and retries on failure
- Logs all execution details to Airflow UI

**Schedule**: `0 2 * * *` (daily at 2:00 AM UTC)

**How to Access**:

```
Open in browser: http://localhost:8080
Login: admin / admin (default)
Navigate to DAGs → recommendation_training_pipeline
```

**Available Actions in Airflow UI**:
- Trigger DAG manually
- View DAG structure
- Check task logs
- Monitor past runs
- Set retry policies

---

## 6. Configuration Scripts

### `src/config/loader.py`

**Purpose**: Load and resolve environment-specific configuration

**Features**:
- Loads YAML configuration from `configs/config.yaml`
- Resolves environment variables like `${MLFLOW_TRACKING_URI}`
- Supports default values like `${VAR:default_value}`
- Centralized config for entire application

**Usage**:

```python
from src.config.loader import ConfigLoader

config = ConfigLoader()

# Get values
tracking_uri = config.get("mlflow.tracking_uri")
experiment_name = config.get("mlflow.experiment_name")
model_path = config.get("model.save_path")

# Get with default
batch_size = config.get("training.batch_size", default=32)
```

**Environment Resolution**:
```yaml
mlflow:
  tracking_uri: ${MLFLOW_TRACKING_URI:http://localhost:5000}
  # Uses env var MLFLOW_TRACKING_URI, falls back to http://localhost:5000
```

---

## 7. Logging Script

### `src/logging/logger.py`

**Purpose**: Centralized logging configuration

**Features**:
- Logs to console and file
- Different log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Includes timestamp, logger name, and level

**Usage**:

```python
from src.logging.logger import get_logger

logger = get_logger(__name__)

logger.info("Training started")
logger.warning("Low data quality")
logger.error("Model failed to train")
```

---

## Script Execution Flow

```
┌─────────────────────────────────────────┐
│  User Action (Development/Deployment)   │
└──────────────┬──────────────────────────┘
               │
        ┌──────▼──────────┐
        │  Local Testing  │
        │   (Optional)    │
        └──────┬──────────┘
               │
        ┌──────▼────────────────────────┐
        │ validate_deployment.py         │
        │ (Check prerequisites)          │
        └──────┬────────────────────────┘
               │
        ┌──────▼──────────────────────────┐
        │ Training Pipeline               │
        │ - Data Ingestion               │
        │ - Feature Engineering          │
        │ - Model Training               │
        │ - Registry & Promotion         │
        └──────┬──────────────────────────┘
               │
        ┌──────▼──────────────────────────┐
        │ API Service                      │
        │ - Health Check                  │
        │ - Metrics Endpoint              │
        │ - Recommendation Endpoint       │
        └──────┬──────────────────────────┘
               │
        ┌──────▼──────────────────────────┐
        │ deploy_checklist.sh              │
        │ (Final validation)               │
        └──────┬──────────────────────────┘
               │
        ┌──────▼──────────────────────────┐
        │ Production Deployment            │
        │ (docker-compose up -d)           │
        └──────────────────────────────────┘
```

---

## Common Script Combinations

### Scenario 1: Quick Local Development Test
```powershell
# 1. Start services
docker compose up -d

# 2. Wait for services
Start-Sleep -Seconds 15

# 3. Test health
curl http://localhost:8000/health

# 4. Run validation
docker compose exec api python scripts/validate_deployment.py

# 5. Stop services
docker compose down
```

### Scenario 2: Full Deployment Pipeline
```powershell
# 1. Validate everything
python scripts/validate_deployment.py

# 2. Run deployment checklist
bash scripts/deploy_checklist.sh

# 3. Build and start
docker compose up -d --build

# 4. Train model
docker compose exec api python src/pipelines/training_pipeline.py

# 5. Test endpoints
# Use Postman collection

# 6. Monitor
docker compose logs -f api
```

### Scenario 3: Airflow-Based Training (Scheduled)
```powershell
# 1. Start stack with Airflow
docker compose up -d

# 2. Open Airflow UI
Start-Process "http://localhost:8080"

# 3. Find DAG: recommendation_training_pipeline
# 4. Click "Trigger DAG" button
# 5. Monitor task execution in Airflow UI
```

---

## Debugging Scripts

### View Script Logs
```powershell
# API logs
docker compose logs api -f

# MLflow logs
docker compose logs mlflow -f

# Airflow logs
docker compose logs airflow-scheduler -f

# All logs
docker compose logs -f
```

### Test Individual Components
```powershell
# Test config loading
docker compose exec api python -c "from src.config.loader import ConfigLoader; c = ConfigLoader(); print(c.get('project.name'))"

# Test inference
docker compose exec api python -c "from src.inference.predict import load_model; print('Inference OK')"

# Test API manually
docker compose exec api python -m uvicorn src.api.main:app --host 0.0.0.0
```

---

## Summary Table

| Script | Purpose | When to Use | Run Command |
|--------|---------|-------------|------------|
| `validate_deployment.py` | Pre-deployment checks | Before deploying | `python scripts/validate_deployment.py` |
| `deploy_checklist.sh` | Final deployment validation | Before prod deploy | `bash scripts/deploy_checklist.sh` |
| `training_pipeline.py` | Train models end-to-end | Train or retrain | `python src/pipelines/training_pipeline.py` |
| `recommendation_training_dag.py` | Scheduled training | Daily automation | Trigger in Airflow UI |
| `validate.py` | Data quality checks | After data load | Called by training pipeline |
| `config/loader.py` | Config management | Throughout app | Imported by modules |
| `logger.py` | Application logging | Throughout app | Imported by modules |

