# Local Setup and Testing Guide

## Prerequisites Check

Before starting, ensure you have:
- Docker Desktop installed and running
- Python 3.11+ installed
- Git installed
- Postman installed (for API testing)

## Step 1: Navigate to Project Directory

```powershell
cd c:\Users\mdrud\mlops_projects\mlops_recommendation_sysytem
```

## Step 2: Verify Environment Configuration

The project uses environment variables. Three profiles are available:

```powershell
# Check available env files
ls -Name .env*

# Current development .env should be active
cat .env
```

Output should show:
```
PROJECT_ENV=dev
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_TRACKING_URI_DOCKER=http://mlflow:5000
MLFLOW_EXPERIMENT_NAME=course_recommendation.dev
...
```

## Step 3: Start the Complete Stack (Recommended)

This starts API, MLflow, and Airflow in Docker Compose:

```powershell
# Build and start all services
docker compose up --build

# Or start in background
docker compose up -d --build
```

Expected output:
```
 Building 12.3s (15/15)
 Creating course-recommendation-api
  Creating course-recommendation-mlflow
  Creating course-recommendation-airflow-webserver
  Creating course-recommendation-airflow-scheduler
```

Check status:
```powershell
docker compose ps
```

Expected:
```
NAME                                  STATUS
course-recommendation-api             Up 2 seconds
course-recommendation-mlflow          Up 2 seconds
course-recommendation-airflow-webserver    Up 2 seconds
course-recommendation-airflow-scheduler    Up 2 seconds
```

## Step 4: Verify Services Are Running

Check each service:

```powershell
# API health check
curl http://localhost:8000/health

# MLflow server
Start-Process "http://localhost:5000"

# Airflow UI
Start-Process "http://localhost:8080"
```

---

## POSTMAN API Testing

### Import Collection

Create a new Postman Collection:

1. **Collection Name**: `Course Recommendation API`
2. **Base URL**: `http://localhost:8000`

### Test Cases

#### 1. Health Check

**Endpoint**: `GET /health`

**No request body needed**

```
GET http://localhost:8000/health
```

**Expected Response (200 OK)**:
```json
{
  "status": "ok",
  "service": "course-recommendation-api",
  "timestamp": "2026-08-31T16:45:23.123456",
  "version": "1.0.0",
  "environment": "dev"
}
```

---

#### 2. Get Metrics

**Endpoint**: `GET /metrics`

**No request body needed**

```
GET http://localhost:8000/metrics
```

**Expected Response (200 OK)**:
```json
{
  "requests_total": 5,
  "requests_success": 4,
  "requests_error": 1,
  "avg_response_time_ms": 245.5
}
```

---

#### 3. Content-Based Filtering (RandomForest) with Features

**Endpoint**: `POST /recommend/content-based`

**Important**: This endpoint requires **actual feature data** for the candidate courses. The model uses:
- **Numerical features**: course_duration_hours, enrollment_numbers, course_price, feedback_score, time_spent_hours, previous_courses_taken
- **Categorical features**: category, difficulty_level

**Request Body (Postman)**:
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

**Expected Response (200 OK)**:
```json
{
  "user_id": 5,
  "top_n": 5,
  "model_type": "content_based_random_forest",
  "recommendations": [
    {
      "rank": 1,
      "user_id": 5,
      "course_id": 102,
      "score": 4.85,
      "model_type": "content_based_random_forest"
    },
    {
      "rank": 2,
      "user_id": 5,
      "course_id": 105,
      "score": 4.72,
      "model_type": "content_based_random_forest"
    },
    {
      "rank": 3,
      "user_id": 5,
      "course_id": 101,
      "score": 4.61,
      "model_type": "content_based_random_forest"
    },
    {
      "rank": 4,
      "user_id": 5,
      "course_id": 104,
      "score": 4.52,
      "model_type": "content_based_random_forest"
    },
    {
      "rank": 5,
      "user_id": 5,
      "course_id": 103,
      "score": 4.38,
      "model_type": "content_based_random_forest"
    }
  ]
}
```

---

#### 4. Collaborative Filtering (SVD) without Features

**Endpoint**: `POST /recommend/collaborative`

**Important**: This endpoint **ONLY needs course IDs**, no feature data required.

**Request Body (Postman)**:
```json
{
  "user_id": 5,
  "top_n": 5,
  "model_type": "collaborative_filtering",
  "candidate_course_ids": [101, 102, 103, 104, 105, 106, 107, 108]
}
```

**Expected Response (200 OK)**:
```json
{
  "user_id": 5,
  "top_n": 5,
  "model_type": "collaborative_filtering",
  "recommendations": [
    {
      "rank": 1,
      "user_id": 5,
      "course_id": 107,
      "score": 4.8,
      "model_type": "collaborative_filtering"
    },
    {
      "rank": 2,
      "user_id": 5,
      "course_id": 105,
      "score": 4.6,
      "model_type": "collaborative_filtering"
    },
    {
      "rank": 3,
      "user_id": 5,
      "course_id": 108,
      "score": 4.5,
      "model_type": "collaborative_filtering"
    },
    {
      "rank": 4,
      "user_id": 5,
      "course_id": 103,
      "score": 4.3,
      "model_type": "collaborative_filtering"
    },
    {
      "rank": 5,
      "user_id": 5,
      "course_id": 106,
      "score": 4.1,
      "model_type": "collaborative_filtering"
    }
  ]
}
```

---

### Postman Collection JSON (Import Ready)

Copy and import this into Postman:

```json
{
  "info": {
    "name": "Course Recommendation API",
    "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
  },
  "item": [
    {
      "name": "Health Check",
      "request": {
        "method": "GET",
        "url": {
          "raw": "{{base_url}}/health",
          "host": ["{{base_url}}"],
          "path": ["health"]
        }
      }
    },
    {
      "name": "Get Metrics",
      "request": {
        "method": "GET",
        "url": {
          "raw": "{{base_url}}/metrics",
          "host": ["{{base_url}}"],
          "path": ["metrics"]
        }
      }
    },
    {
      "name": "Recommend (Content-Based)",
      "request": {
        "method": "POST",
        "header": [{"key": "Content-Type", "value": "application/json"}],
        "body": {
          "mode": "raw",
          "raw": "{\n  \"user_id\": 5,\n  \"top_n\": 5,\n  \"model_type\": \"content_based_random_forest\",\n  \"course_duration_hours\": [20.5, 15.0, 30.0, 25.5, 18.0],\n  \"enrollment_numbers\": [1500, 2300, 890, 1200, 3400],\n  \"course_price\": [49.99, 99.99, 29.99, 79.99, 149.99],\n  \"feedback_score\": [4.5, 4.8, 3.9, 4.6, 4.7],\n  \"time_spent_hours\": [15.0, 12.5, 20.0, 18.5, 14.0],\n  \"previous_courses_taken\": [5, 8, 3, 6, 10],\n  \"category\": [\"Python\", \"Data Science\", \"Web Development\", \"Machine Learning\", \"Python\"],\n  \"difficulty_level\": [\"Beginner\", \"Intermediate\", \"Beginner\", \"Intermediate\", \"Advanced\"]\n}"
        },
        "url": {
          "raw": "{{base_url}}/recommend/content-based",
          "host": ["{{base_url}}"],
          "path": ["recommend", "content-based"]
        }
      }
    },
    {
      "name": "Recommend (Collaborative Filtering)",
      "request": {
        "method": "POST",
        "header": [{"key": "Content-Type", "value": "application/json"}],
        "body": {
          "mode": "raw",
          "raw": "{\n  \"user_id\": 5,\n  \"top_n\": 5,\n  \"model_type\": \"collaborative_filtering\",\n  \"candidate_course_ids\": [101, 102, 103, 104, 105, 106, 107, 108]\n}"
        },
        "url": {
          "raw": "{{base_url}}/recommend/collaborative",
          "host": ["{{base_url}}"],
          "path": ["recommend", "collaborative"]
        }
      }
    }
  ],
  "variable": [
    {
      "key": "base_url",
      "value": "http://localhost:8000"
    }
  ]
}
```

---

### Testing via cURL (PowerShell)

**Health Check**:
```powershell
curl -Uri "http://localhost:8000/health" -Method GET
```

**Content-Based Filtering**:
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

curl -Uri "http://localhost:8000/recommend/content-based" -Method POST -Headers @{"Content-Type"="application/json"} -Body $body
```

**Collaborative Filtering**:
```powershell
$body = @{
    user_id = 5
    top_n = 5
    model_type = "collaborative_filtering"
    candidate_course_ids = @(101, 102, 103, 104, 105, 106, 107, 108)
} | ConvertTo-Json

curl -Uri "http://localhost:8000/recommend/collaborative" -Method POST -Headers @{"Content-Type"="application/json"} -Body $body
```

---

### Error Responses

**Model File Not Found (404)**:
```json
{
  "detail": "Model file not found: models/recommender_model.joblib"
}
```

**Missing Required Features (400)**:
```json
{
  "detail": "Content-based model bundle is missing the trained preprocessor and feature columns."
}
```

**Internal Server Error (500)**:
```json
{
  "detail": "Internal server error"
}
```

#### 1. Health Check

```
GET http://localhost:8000/health

Response (200 OK):
{
  "status": "ok",
  "service": "course-recommendation-api",
  "timestamp": "2026-08-31T15:30:00.123456",
  "version": "1.0.0",
  "environment": "dev"
}
```

In Postman:
- Method: **GET**
- URL: `{{base_url}}/health`
- Headers: None
- Body: None

#### 2. Get Metrics

```
GET http://localhost:8000/metrics

Response (200 OK):
{
  "requests_total": 5,
  "requests_success": 4,
  "requests_error": 1,
  "avg_response_time_ms": 45.5
}
```

In Postman:
- Method: **GET**
- URL: `{{base_url}}/metrics`
- Headers: None
- Body: None

#### 3. Content-Based Recommendation

```
POST http://localhost:8000/recommend
Content-Type: application/json

Request Body:
{
  "user_id": 1,
  "top_n": 5,
  "model_type": "content_based_random_forest",
  "candidate_features": [
    {
      "course_id": 101,
      "course_duration_hours": 20,
      "enrollment_numbers": 150,
      "course_price": 49.99,
      "rating": 4.5
    },
    {
      "course_id": 102,
      "course_duration_hours": 30,
      "enrollment_numbers": 200,
      "course_price": 79.99,
      "rating": 4.7
    },
    {
      "course_id": 103,
      "course_duration_hours": 15,
      "enrollment_numbers": 100,
      "course_price": 29.99,
      "rating": 4.2
    }
  ],
  "candidate_course_ids": []
}

Response (200 OK):
{
  "user_id": 1,
  "top_n": 5,
  "model_type": "content_based_random_forest",
  "recommendations": [
    {
      "rank": 1,
      "user_id": 1,
      "course_id": 102,
      "score": 4.7,
      "model_type": "content_based_random_forest"
    },
    {
      "rank": 2,
      "user_id": 1,
      "course_id": 101,
      "score": 4.5,
      "model_type": "content_based_random_forest"
    },
    {
      "rank": 3,
      "user_id": 1,
      "course_id": 103,
      "score": 4.2,
      "model_based_random_forest"
    }
  ]
}
```

In Postman:
- Method: **POST**
- URL: `{{base_url}}/recommend`
- Headers: `Content-Type: application/json`
- Body (raw JSON):
```json
{
  "user_id": 1,
  "top_n": 5,
  "model_type": "content_based_random_forest",
  "candidate_features": [
    {
      "course_id": 101,
      "course_duration_hours": 20,
      "enrollment_numbers": 150,
      "course_price": 49.99,
      "rating": 4.5
    }
  ],
  "candidate_course_ids": []
}
```

#### 4. Collaborative Filtering Recommendation

```
POST http://localhost:8000/recommend
Content-Type: application/json

Request Body:
{
  "user_id": 5,
  "top_n": 10,
  "model_type": "collaborative_filtering",
  "candidate_features": [],
  "candidate_course_ids": [101, 102, 103, 104, 105, 106, 107, 108]
}

Response (200 OK):
{
  "user_id": 5,
  "top_n": 10,
  "model_type": "collaborative_filtering",
  "recommendations": [
    {
      "rank": 1,
      "user_id": 5,
      "course_id": 107,
      "score": 4.8,
      "model_type": "collaborative_filtering"
    },
    {
      "rank": 2,
      "user_id": 5,
      "course_id": 105,
      "score": 4.6,
      "model_type": "collaborative_filtering"
    }
  ]
}
```

In Postman:
- Method: **POST**
- URL: `{{base_url}}/recommend`
- Headers: `Content-Type: application/json`
- Body (raw JSON):
```json
{
  "user_id": 5,
  "top_n": 10,
  "model_type": "collaborative_filtering",
  "candidate_features": [],
  "candidate_course_ids": [101, 102, 103, 104, 105, 106, 107, 108]
}
```

### Postman Environment Variables

Create a Postman Environment:

```json
{
  "name": "Local Development",
  "values": [
    {
      "key": "base_url",
      "value": "http://localhost:8000",
      "enabled": true
    },
    {
      "key": "mlflow_url",
      "value": "http://localhost:5000",
      "enabled": true
    },
    {
      "key": "airflow_url",
      "value": "http://localhost:8080",
      "enabled": true
    }
  ]
}
```

---

## Local Development (Without Docker - Optional)

If you prefer running locally without Docker:

```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Export environment variables
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
$env:MLFLOW_EXPERIMENT_NAME = "course_recommendation.dev"
$env:PROJECT_ENV = "dev"

# Start MLflow in separate terminal
mlflow server --host 0.0.0.0 --port 5000

# Start API in another terminal
python -m uvicorn src.api.main:app --reload

# In a third terminal, run training
python src/pipelines/training_pipeline.py
```

---

## Stopping Services

```powershell
# Stop all containers
docker compose down

# Stop with volume cleanup
docker compose down -v

# View logs
docker compose logs -f api
docker compose logs -f mlflow
```

---

## Troubleshooting

### Port Already in Use

If port 8000, 5000, or 8080 is already in use:

```powershell
# Find process using port 8000
netstat -ano | findstr :8000

# Kill process (replace PID)
taskkill /PID <PID> /F
```

### Docker Not Running

```powershell
# Start Docker Desktop
Start-Service docker

# Or restart Docker daemon
Restart-Service docker
```

### No Model Artifact

If API returns 404 model not found:

```powershell
# Train the model first
docker compose exec api python src/pipelines/training_pipeline.py

# Or train locally
python src/pipelines/training_pipeline.py
```

### Connection Refused

Wait 10-15 seconds after starting services for all containers to initialize:

```powershell
# Check logs
docker compose logs

# Wait and retry
Start-Sleep -Seconds 15
curl http://localhost:8000/health
```

