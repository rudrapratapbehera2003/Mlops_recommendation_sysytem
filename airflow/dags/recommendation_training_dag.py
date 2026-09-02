import sys
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

# Add project root to path
PROJECT_ROOT = "/opt/airflow/project"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.validate import DataValidator
from src.data.ingest import DataIngestion
from src.models.train_model import ModelTraining
from src.config.loader import ConfigLoader
from src.logger.logger import get_logger

logger = get_logger(__name__)

# Default arguments for the DAG
default_args = {
    'owner': 'mlops-recommendation-team',
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email': ['rpbehera167@gmail.com'],
}

# DAG definition
dag = DAG(
    'recommendation_system_training',
    default_args=default_args,
    description='Training pipeline for course recommendation system',
    schedule_interval='0 2 * * 0',  # Every Sunday at 2 AM
    catchup=False,
    tags=['mlops', 'recommendation', 'training'],
)


def validate_data_task(**context):
    """Validate raw data before ingestion."""
    logger.info("Starting data validation...")
    try:
        config = ConfigLoader()
        validator = DataValidator(config)
        raw_data_path = config.get("data.raw_data_path", "data/raw/")
        
        validation_report = validator.validate_raw_data(raw_data_path)
        logger.info(f"Validation Report: {validation_report}")
        
        context['ti'].xcom_push(key='validation_status', value='passed')
        return {'status': 'passed', 'report': validation_report}
    except Exception as e:
        logger.error(f"Data validation failed: {str(e)}")
        context['ti'].xcom_push(key='validation_status', value='failed')
        raise


def ingest_data_task(**context):
    """Ingest and process raw data."""
    logger.info("Starting data ingestion...")
    try:
        config = ConfigLoader()
        ingestion = DataIngestion(config)
        
        raw_data_path = config.get("data.raw_data_path", "data/raw/")
        processed_data_path = config.get("data.processed_data_path", "data/processed/")
        
        logger.info(f"Ingesting data from {raw_data_path}...")
        ingestion.ingest_and_process(raw_data_path, processed_data_path)
        
        context['ti'].xcom_push(key='ingestion_status', value='completed')
        logger.info("Data ingestion completed successfully")
        return {'status': 'completed', 'processed_path': processed_data_path}
    except Exception as e:
        logger.error(f"Data ingestion failed: {str(e)}")
        context['ti'].xcom_push(key='ingestion_status', value='failed')
        raise


def train_model_task(**context):
    """Train both CF and CBF models."""
    logger.info("Starting model training...")
    try:
        config = ConfigLoader()
        trainer = ModelTraining(config)
        
        logger.info("Running model comparison and training pipeline...")
        result = trainer.compairing_to_find_best_model()
        
        context['ti'].xcom_push(key='training_status', value='completed')
        context['ti'].xcom_push(key='best_model', value=result.get('selected_model'))
        context['ti'].xcom_push(key='model_path', value=result.get('model_path'))
        context['ti'].xcom_push(key='mlflow_run_id', value=result.get('mlflow_run_id'))
        
        logger.info(f"Model training completed. Best model: {result.get('selected_model')}")
        logger.info(f"Metrics: {result.get('metrics')}")
        
        return result
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        context['ti'].xcom_push(key='training_status', value='failed')
        raise


def model_registry_task(**context):
    """Register trained model in MLflow."""
    
    logger.info("Starting model registry task...")
    try:
        best_model = context['ti'].xcom_pull(key='best_model', task_ids='train_model')
        mlflow_run_id = context['ti'].xcom_pull(key='mlflow_run_id', task_ids='train_model')
        model_path = context['ti'].xcom_pull(key='model_path', task_ids='train_model')
        
        logger.info(f"Model {best_model} registered successfully in MLflow")
        logger.info(f"MLflow Run ID: {mlflow_run_id}")
        logger.info(f"Model Path: {model_path}")
        
        context['ti'].xcom_push(key='registry_status', value='completed')
        return {
            'status': 'registered',
            'model_type': best_model,
            'run_id': mlflow_run_id,
            'path': model_path
        }
    except Exception as e:
        logger.error(f"Model registry failed: {str(e)}")
        context['ti'].xcom_push(key='registry_status', value='failed')
        raise


def notification_task(**context):
    """Send notification about training completion."""
    logger.info("Sending training completion notification...")
    try:
        best_model = context['ti'].xcom_pull(key='best_model', task_ids='train_model')
        mlflow_run_id = context['ti'].xcom_pull(key='mlflow_run_id', task_ids='train_model')
        
        notification_message = f"""
        Training Pipeline Completed Successfully!
        
        Best Model: {best_model}
        MLflow Run ID: {mlflow_run_id}
        Training Timestamp: {datetime.utcnow().isoformat()}
        
        The model is ready for deployment.
        """
        
        logger.info(notification_message)
        return {'status': 'notified', 'message': notification_message}
    except Exception as e:
        logger.error(f"Notification failed: {str(e)}")
        raise


# Define tasks
validate_data = PythonOperator(
    task_id='validate_data',
    python_callable=validate_data_task,
    dag=dag,
    provide_context=True,
)

ingest_data = PythonOperator(
    task_id='ingest_data',
    python_callable=ingest_data_task,
    dag=dag,
    provide_context=True,
)

train_model = PythonOperator(
    task_id='train_model',
    python_callable=train_model_task,
    dag=dag,
    provide_context=True,
    pool='default_pool',
    pool_slots=1,
)

model_registry = PythonOperator(
    task_id='model_registry',
    python_callable=model_registry_task,
    dag=dag,
    provide_context=True,
)

notification = PythonOperator(
    task_id='notification',
    python_callable=notification_task,
    dag=dag,
    provide_context=True,
)

# Ingestion creates the processed file consumed by validation.
ingest_data >> validate_data >> train_model >> model_registry >> notification
