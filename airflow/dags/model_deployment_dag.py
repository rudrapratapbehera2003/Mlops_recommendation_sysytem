import os
import sys
import json
import urllib.request
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

PROJECT_ROOT = "/opt/airflow/project"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.logger.logger import get_logger

logger = get_logger(__name__)


def _mlflow_client():
    import mlflow
    from mlflow.tracking import MlflowClient

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient(tracking_uri=tracking_uri)


def _api_request(path, method="GET", payload=None):
    api_url = os.getenv("API_URL", "http://api:8000").rstrip("/")
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{api_url}{path}",
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))

# Default arguments
default_args = {
    'owner': 'mlops-recommendation-team',
    'retries': 1,
    'retry_delay': timedelta(minutes=10),
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email': ['rpbehera167@gmail.com']
}

# DAG definition
deployment_dag = DAG(
    'recommendation_model_deployment',
    default_args=default_args,
    description='Model deployment and promotion pipeline',
    schedule_interval=None,  # Triggered manually or by external system
    catchup=False,
    tags=['mlops', 'deployment'],
)


def check_model_quality_task(**context):
    logger.info("Checking model quality metrics...")
    from src.config.loader import ConfigLoader

    config = ConfigLoader()
    client = _mlflow_client()
    experiment_name = config.get("mlflow.experiment_name", "course_recommendation.dev")
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"MLflow experiment does not exist: {experiment_name}")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attribute.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise ValueError("No completed MLflow training run was found.")

    run = runs[0]
    metrics = run.data.metrics
    thresholds = {
        "cbf_rmse": 1.0,
        "cbf_mae": 0.8,
        "cf_rmse": 1.2,
        "cf_mae": 1.0,
    }
    missing = [name for name in thresholds if name not in metrics]
    if missing:
        raise ValueError(f"Latest MLflow run is missing metrics: {missing}")
    failed = {
        name: metrics[name]
        for name, threshold in thresholds.items()
        if metrics[name] > threshold
    }
    if failed:
        raise ValueError(f"Model quality thresholds failed: {failed}")

    context["ti"].xcom_push(key="run_id", value=run.info.run_id)
    context["ti"].xcom_push(key="quality_check", value="passed")
    logger.info("Quality check passed for MLflow run %s: %s", run.info.run_id, metrics)
    return {"status": "passed", "run_id": run.info.run_id, "metrics": metrics}


def run_integration_tests_task(**context):

    logger.info("Running integration tests...")
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error(result.stdout)
        logger.error(result.stderr)
        raise RuntimeError("Project integration tests failed.")
    context["ti"].xcom_push(key="integration_tests", value="passed")
    return {"status": "passed", "output": result.stdout[-2000:]}


def deploy_to_staging_task(**context):
    """Deploy model to staging environment."""
    logger.info("Deploying model to staging...")
    from src.config.loader import ConfigLoader

    config = ConfigLoader()
    model_name = config.get("mlflow.model_registry_name", "course_recommendation")
    run_id = context["ti"].xcom_pull(key="run_id", task_ids="check_model_quality")
    client = _mlflow_client()
    versions = client.search_model_versions(f"name='{model_name}'")
    matching = [version for version in versions if version.run_id == run_id]
    if not matching:
        raise ValueError(f"No registered version found for MLflow run {run_id}.")
    version = max(matching, key=lambda item: int(item.version))
    if version.current_stage.lower() != "staging":
        raise ValueError(
            f"Model version {version.version} is not in Staging: {version.current_stage}"
        )
    context["ti"].xcom_push(key="model_version", value=version.version)
    context["ti"].xcom_push(key="staging_deployment", value="completed")
    return {"status": "deployed", "model_name": model_name, "version": version.version}


def run_smoke_tests_task(**context):
    """Run smoke tests against staging deployment."""
    logger.info("Running smoke tests on staging...")
    health = _api_request("/health")
    if health.get("status") != "ok":
        raise RuntimeError(f"API health check failed: {health}")
    recommendation = _api_request(
        "/recommend/content-based",
        method="POST",
        payload={"user_id": 15796, "top_n": 1},
    )
    if not isinstance(recommendation.get("recommendations"), list):
        raise RuntimeError("Recommendation smoke test returned an invalid payload.")
    context["ti"].xcom_push(key="smoke_tests", value="passed")
    return {"status": "passed", "health": health, "recommendation": recommendation}


def deploy_to_production_task(**context):
    """Deploy model to production environment."""
    logger.info("Deploying model to production...")
    smoke_status = context["ti"].xcom_pull(key="smoke_tests", task_ids="run_smoke_tests")
    if smoke_status != "passed":
        raise ValueError("Production promotion blocked because smoke tests did not pass.")
    context["ti"].xcom_push(key="production_deployment", value="completed")
    return {"status": "promoted", "model_version": context["ti"].xcom_pull(key="model_version", task_ids="deploy_to_staging")}


def monitor_model_performance_task(**context):
    """Monitor deployed model performance."""
    logger.info("Monitoring model performance...")
    metrics = _api_request("/metrics")
    logger.info("API performance metrics: %s", metrics)
    return {"status": "monitored", "metrics": metrics}


# Define tasks
check_quality = PythonOperator(
    task_id='check_model_quality',
    python_callable=check_model_quality_task,
    dag=deployment_dag,
    provide_context=True,
)

run_tests = PythonOperator(
    task_id='run_integration_tests',
    python_callable=run_integration_tests_task,
    dag=deployment_dag,
    provide_context=True,
)

deploy_staging = PythonOperator(
    task_id='deploy_to_staging',
    python_callable=deploy_to_staging_task,
    dag=deployment_dag,
    provide_context=True,
)

smoke_tests = PythonOperator(
    task_id='run_smoke_tests',
    python_callable=run_smoke_tests_task,
    dag=deployment_dag,
    provide_context=True,
)

deploy_prod = PythonOperator(
    task_id='deploy_to_production',
    python_callable=deploy_to_production_task,
    dag=deployment_dag,
    provide_context=True,
)

monitor = PythonOperator(
    task_id='monitor_performance',
    python_callable=monitor_model_performance_task,
    dag=deployment_dag,
    provide_context=True,
)

# Set task dependencies
check_quality >> run_tests >> deploy_staging >> smoke_tests >> deploy_prod >> monitor
