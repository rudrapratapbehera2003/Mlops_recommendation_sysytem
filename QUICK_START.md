# Quick Start Guide - Run Project Locally

## Prerequisites
- Docker Desktop (installed and running)
- Python 3.11+
- Git
- Postman (for API testing)

---

## Step 1: Navigate to Project

```powershell
cd c:\Users\mdrud\mlops_projects\mlops_recommendation_sysytem
```

---

## Step 2: Start the Complete Stack (Recommended)

```powershell
# Build and start all services (API, MLflow, Airflow)
docker compose up --build

# Or start in background
docker compose up -d --build
```

**Wait for services to start:**
- API: http://localhost:8000/docs (Swagger UI)
- MLflow: http://localhost:5000
- Airflow: http://localhost:8080

Verify services are running:
```powershell
docker compose ps
```

---

## Step 3: Train the Model (Optional - Only if needed)

```powershell
# Run training inside the API container
docker compose exec api python src/pipelines/training_pipeline.py

# Or run training locally
python src/pipelines/training_pipeline.py
```

**Expected Output:**
- Data ingestion ✓
- Data validation ✓
- Feature engineering ✓
- Model training (CF + CBF) ✓
- Model registered in MLflow ✓

---

## Step 4: Test API Endpoints

### Quick Health Check

```powershell
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "ok",
  "service": "course-recommendation-api",
  "timestamp": "2026-08-31T...",
  "version": "1.0.0",
  "environment": "dev"
}
```

### Test Collaborative Filtering Endpoint

```powershell
$body = @{
    user_id = 5
    top_n = 5
    model_type = "collaborative_filtering"
    candidate_course_ids = @(101, 102, 103, 104, 105, 106, 107, 108)
} | ConvertTo-Json

curl -Uri "http://localhost:8000/recommend/collaborative" `
  -Method POST `
  -Headers @{"Content-Type"="application/json"} `
  -Body $body
```

### Test Content-Based Filtering Endpoint (WITH FEATURES)

```powershell
$body = @{
    user_id = 5
    top_n = 5
    model_type = "content_based_random_forest"
    course_duration_hours = @(20.5, 15.0, 30.0, 25.5, 18.0)
    enrollment_numbers = @(1500, 2300, 890, 1200, 3400)
    course_price = @(49.99, 99.99, 29.99, 79.99, 149.99)
    feedback_score = @(4.5, 4.8, 3.9, 4.6, 4.7)
    time_spent_hours = @(15.0, 12.5, 20.0, 18.5, 14.0)
    previous_courses_taken = @(5, 8, 3, 6, 10)
    category = @("Python", "Data Science", "Web Development", "Machine Learning", "Python")
    difficulty_level = @("Beginner", "Intermediate", "Beginner", "Intermediate", "Advanced")
} | ConvertTo-Json

curl -Uri "http://localhost:8000/recommend/content-based" `
  -Method POST `
  -Headers @{"Content-Type"="application/json"} `
  -Body $body
```

---

## Step 5: Using Postman (Recommended for Testing)

1. **Open Postman**
2. **Create New Collection**: "Course Recommendation API"
3. **Set Base URL**: `http://localhost:8000`

### Add These Requests:

#### Request 1: Health Check
- **Method**: GET
- **URL**: `{{base_url}}/health`
- **Body**: None

#### Request 2: Collaborative Filtering
- **Method**: POST
- **URL**: `{{base_url}}/recommend/collaborative`
- **Body** (JSON):
```json
{
  "user_id": 5,
  "top_n": 5,
  "model_type": "collaborative_filtering",
  "candidate_course_ids": [101, 102, 103, 104, 105, 106, 107, 108]
}
```

#### Request 3: Content-Based Filtering (With ALL Features)
- **Method**: POST
- **URL**: `{{base_url}}/recommend/content-based`
- **Body** (JSON):
```json
{
  "user_id": 5,
  "top_n": 5,
  "model_type": "content_based_random_forest",
  "course_duration_hours": [20.5, 15.0, 30.0, 25.5, 18.0],
  "enrollment_numbers": [1500, 2300, 890, 1200, 3400],
  "course_price": [49.99, 99.99, 29.99, 79.99, 149.99],
  "feedback_score": [4.5, 4.8, 3.9, 4.6, 4.7],
  "time_spent_hours": [15.0, 12.5, 20.0, 18.5, 14.0],
  "previous_courses_taken": [5, 8, 3, 6, 10],
  "category": ["Python", "Data Science", "Web Development", "Machine Learning", "Python"],
  "difficulty_level": ["Beginner", "Intermediate", "Beginner", "Intermediate", "Advanced"]
}
```

#### Request 4: Metrics
- **Method**: GET
- **URL**: `{{base_url}}/metrics`
- **Body**: None

---

## Step 6: View MLflow & Airflow UIs

### MLflow (Model Registry & Experiment Tracking)
```powershell
Start-Process "http://localhost:5000"
```
- View trained models
- Check experiments
- See model registry and stages (Staging/Production)

### Airflow (Orchestration)
```powershell
Start-Process "http://localhost:8080"
```
- Monitor training DAG
- Trigger manual runs
- View logs and execution history

---

## Step 7: API Documentation

Open Swagger UI for interactive API docs:
```
http://localhost:8000/docs
```

Features:
- Try endpoints directly in browser
- View request/response schemas
- See error examples

---

## Run Tests Locally

```powershell
# All tests
python -m pytest tests/ -v

# Specific test file
python -m pytest tests/test_inference_predict.py -v

# With coverage
python -m pytest tests/ --cov=src
```

---

## Stop Services

```powershell
# Stop all services
docker compose down

# Stop and remove volumes
docker compose down -v

# View logs
docker compose logs -f api

# View specific service logs
docker compose logs mlflow
```

---

## API Endpoints Summary

| Endpoint | Method | Purpose | Requires Features |
|----------|--------|---------|------------------|
| `/health` | GET | Check API status | No |
| `/metrics` | GET | View API metrics | No |
| `/recommend/collaborative` | POST | Get CF recommendations (SVD) | Only course IDs |
| `/recommend/content-based` | POST | Get CBF recommendations (RandomForest) | YES - All features |

---

## Key Differences Between Models

### Collaborative Filtering (CF)
- **Uses**: User-course interaction history (ratings)
- **Input**: Only `candidate_course_ids`
- **Best for**: Recommending similar courses to what user liked
- **Method**: TruncatedSVD (matrix factorization)

### Content-Based Filtering (CBF)
- **Uses**: Course features (duration, price, category, difficulty, etc.)
- **Input**: **ALL numerical AND categorical features** for each candidate course
- **Best for**: New users with no history; feature-specific recommendations
- **Method**: RandomForestRegressor

---

## Troubleshooting

**API not responding?**
```powershell
docker compose logs api
```

**Model file missing?**
- Run training: `docker compose exec api python src/pipelines/training_pipeline.py`

**MLflow not accessible?**
- Check service: `docker compose logs mlflow`

**Port already in use?**
- Change in `.env`: `APP_PORT=8001`
- Restart: `docker compose down` then `docker compose up -d`

---

## Project Structure

```
src/
  ├── api/              # FastAPI endpoints
  ├── models/           # Model training & registry
  ├── inference/        # Model loading & prediction
  ├── data/             # Data processing
  ├── pipelines/        # ML pipeline orchestration
  ├── config/           # Configuration loading
  └── logging/          # Logging setup

airflow/
  └── dags/             # Scheduled training DAGs

configs/
  └── config.yaml       # Project configuration

scripts/
  ├── validate_deployment.py    # Pre-deployment checks
  └── deploy_checklist.sh       # Deployment checklist

tests/
  ├── test_inference_predict.py # Inference tests
  └── test_model_registry.py    # Registry tests
```

---

## Next Steps

1. Run `docker compose up --build`
2. Test endpoints with Postman
3. Check MLflow for model versions
4. View Airflow DAG for scheduling
5. Run `python scripts/validate_deployment.py` for pre-deployment check

---

**Ready to go!** 🚀
