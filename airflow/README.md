# Airflow Configuration

This directory contains the Airflow orchestration setup for the MLOps Recommendation System.

## Structure

- `dags/` - Contains all DAG definitions
  - `recommendation_training_dag.py` - Main training pipeline DAG
  - `model_deployment_dag.py` - Model deployment and promotion DAG

- `logs/` - Airflow task logs (auto-generated)
- `db/` - Airflow database files (auto-generated)

## Running Airflow

### Via Docker Compose

```bash
# Start all services (API, MLflow, Airflow)
docker compose up -d

# Access Airflow UI
# http://localhost:8080

# Access MLflow UI
# http://localhost:5000
```

### Environment Variables

Ensure the following are set in `.env`:

```
AIRFLOW__CORE__EXECUTOR=SequentialExecutor
AIRFLOW__CORE__LOAD_EXAMPLES=False
AIRFLOW__CORE__DAGS_FOLDER=/opt/airflow/dags
AIRFLOW__CORE__FERNET_KEY=your-fernet-key-here
PYTHONPATH=/opt/airflow/project
```

## Available DAGs

### 1. recommendation_system_training (recommendation_training_dag.py)

**Schedule**: Every Sunday at 2 AM

**Tasks**:
1. `validate_data` - Validate raw data integrity
2. `ingest_data` - Process and prepare data
3. `train_model` - Train CF and CBF models
4. `model_registry` - Register model in MLflow
5. `notification` - Send completion notification

### 2. recommendation_model_deployment (model_deployment_dag.py)

**Schedule**: Manual trigger

**Tasks**:
1. `check_model_quality` - Validate model metrics
2. `run_integration_tests` - Run integration test suite
3. `deploy_to_staging` - Deploy to staging
4. `run_smoke_tests` - Smoke tests on staging
5. `deploy_to_production` - Deploy to production
6. `monitor_performance` - Monitor production metrics

## Useful Commands

### Check DAG Syntax

```bash
docker exec course-recommendation-airflow-webserver airflow dags list
docker exec course-recommendation-airflow-webserver airflow dags validate recommendation_system_training
```

### Trigger a DAG Run

```bash
docker exec course-recommendation-airflow-webserver airflow dags trigger recommendation_system_training
```

### View Logs

```bash
docker exec course-recommendation-airflow-webserver airflow tasks logs recommendation_system_training train_model 2024-01-01
```

### List Latest Runs

```bash
docker exec course-recommendation-airflow-webserver airflow dags list-runs --dag-id recommendation_system_training
```

## Monitoring

- **Airflow UI**: http://localhost:8080 (Admin/Admin by default)
- **MLflow UI**: http://localhost:5000
- **API Health**: http://localhost:8000/health
- **API Metrics**: http://localhost:8000/metrics
