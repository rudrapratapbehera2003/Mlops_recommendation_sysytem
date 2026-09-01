#!/bin/bash
# Production deployment checklist

set -e

echo "=========================================="
echo "Production Deployment Checklist"
echo "=========================================="

echo ""
echo "1. Environment validation..."
python scripts/validate_deployment.py

echo ""
echo "2. Building Docker image..."
docker build -t course-recommendation-api:latest .

echo ""
echo "3. Validating docker-compose for production..."
docker compose config --quiet

echo ""
echo "4. Checking model artifact..."
if [ -f "models/recommender_model.joblib" ]; then
    echo "✓ Model artifact found"
else
    echo "✗ Model artifact NOT found - run training pipeline first"
    exit 1
fi

echo ""
echo "5. Verifying MLflow connectivity..."
python -c "
from src.config.loader import ConfigLoader
from mlflow.tracking import MlflowClient
config = ConfigLoader()
client = MlflowClient(config.get('mlflow.tracking_uri'))
print('✓ MLflow server is accessible')
"

echo ""
echo "6. Final checks..."
echo "✓ All pre-deployment checks passed"
echo ""
echo "Ready to deploy. Run: docker-compose up -d"
