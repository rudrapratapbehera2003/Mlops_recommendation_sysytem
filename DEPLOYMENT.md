# Project Setup and Deployment Guide

## Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- Git

### Local Development Setup
```bash
# Clone and enter project
git clone https://github.com/rudrapratapbehera2003/Mlops_recommendation_sysytem.git
cd mlops_recommendation_system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your local values
```

### Running Locally

**Option 1: Standalone**
```bash
# Train the model
python src/pipelines/training_pipeline.py

# Start API
uvicorn src.api.main:app --reload

# Check health
curl http://localhost:8000/health

# Get recommendations
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "top_n": 5, "model_type": "content_based_random_forest"}'
```

**Option 2: Docker Compose (Full Stack)**
```bash
# Start all services (API, MLflow, Airflow)
docker compose up --build

# Services will be available at:
# - API: http://localhost:8000/docs
# - MLflow: http://localhost:5000
# - Airflow: http://localhost:8080
```

### Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src

# Run specific test file
pytest tests/test_inference_predict.py -v
```

### Code Quality

```bash
# Format code
black src/ tests/

# Check imports
isort src/ tests/

# Lint
pylint src/

# Security scan
bandit -r src/
```

## Architecture

### Components

1. **Data Layer** (`src/data/`)
   - Data ingestion and validation
   - Feature engineering
   - Preprocessing and transformation

2. **Model Layer** (`src/models/`)
   - Collaborative Filtering (TruncatedSVD)
   - Content-Based Filtering (RandomForestRegressor)
   - Model registry and versioning

3. **Inference Layer** (`src/inference/`)
   - Model loading and dispatching
   - Recommendation generation
   - Bundle management

4. **API Layer** (`src/api/`)
   - FastAPI server
   - Health checks and metrics
   - Request validation

5. **Orchestration** (`airflow/dags/`)
   - Training pipeline DAG
   - Scheduled model retraining

### Environment Configuration

Three environment configurations available:

- **Dev** (`.env.dev`): Local development
- **Staging** (`.env.staging`): Pre-production testing
- **Production** (`.env.prod`): Production deployment

Switch environments:
```bash
cp .env.staging .env
docker compose up  # Will use updated config
```

## Deployment

### Pre-Deployment Validation

```bash
# Run comprehensive deployment checks
python scripts/validate_deployment.py

# Run deployment checklist
bash scripts/deploy_checklist.sh
```

### Docker Deployment

```bash
# Build and push image
docker build -t course-recommendation-api:v1.0.0 .
docker push <registry>/course-recommendation-api:v1.0.0

# Deploy with docker-compose
docker compose up -d

# Check status
docker compose ps
docker compose logs api
```

## Monitoring

### Health Endpoints

```bash
# API Health
curl http://localhost:8000/health

# Metrics
curl http://localhost:8000/metrics
```

### MLflow UI

- Navigate to http://localhost:5000
- View experiments, runs, and model registry

### Airflow UI

- Navigate to http://localhost:8080
- Monitor DAG runs and task logs

## CI/CD Pipeline

GitHub Actions workflow (`.github/workflows/ci-cd.yml`) runs on every push:

1. Lint and format checks
2. Unit tests with coverage
3. Docker image build
4. Security scanning
5. Config validation

## Troubleshooting

### Model Not Found
```bash
# Train a new model
python src/pipelines/training_pipeline.py
# or trigger Airflow DAG
```

### API Not Responding
```bash
# Check if container is running
docker compose ps

# View logs
docker compose logs api

# Restart service
docker compose restart api
```

### MLflow Connection Error
```bash
# Ensure MLflow container is running
docker compose logs mlflow

# Check tracking URI in .env
cat .env | grep MLFLOW_TRACKING_URI
```

## Performance Tips

1. Use content-based model for feature-rich data
2. Use collaborative filtering for cold-start problems
3. Cache preprocessor in API container
4. Monitor model drift regularly
5. Retrain when metrics degrade

## Security Considerations

1. Never commit `.env` files with secrets
2. Use a secrets manager in production
3. Enable HTTPS for API endpoints
4. Set strong Fernet keys for Airflow
5. Restrict MLflow UI access in production

## Contributing

1. Create feature branch
2. Make changes and run tests
3. Ensure code passes CI/CD
4. Create pull request
5. After review, merge to main

## Support

For issues or questions:
1. Check logs: `docker compose logs <service>`
2. Review error traces
3. Consult troubleshooting section
4. Open GitHub issue with details
