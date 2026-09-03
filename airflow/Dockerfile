FROM apache/airflow:2.9.1-python3.11

RUN pip install --no-cache-dir \
    'mlflow>=2.14.0,<3.0.0' \
    'pandas>=2.0.0,<2.3.0' \
    'numpy>=1.9.0,<2.0.0' \
    'scipy>=1.11.0,<2.0.0' \
    'scikit-learn>=1.3.0,<1.5.0' \
    'joblib>=1.3.0,<2.0.0' \
    'pyyaml>=6.0.0' \
    'openpyxl>=3.0.0' \
    'python-dotenv==1.0.1' \
    pytest
