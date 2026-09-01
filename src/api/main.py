from typing import Any, Literal, Optional
import os
import time
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator

from src.inference.predict import predict_recommendations

app = FastAPI(
    title="Course Recommendation API",
    version="1.0.0",
    description="API for serving recommendations from the trained recommendation model.",
)


class ContentBasedFeaturesRequest(BaseModel):
    """Features required for Content-Based Filtering (RandomForest)"""
    user_id: int = Field(default=15796, description="User ID to generate recommendations for.", json_schema_extra={"example": 15796})
    top_n: int = Field(default=5, ge=1, le=20, description="Number of recommendations to return.", json_schema_extra={"example": 5})
    model_type: Literal["content_based_random_forest"] = Field(
        default="content_based_random_forest",
        description="Content-based model using RandomForest.",
    )
    
    # Required for all features
    course_ids: list[int] = Field(
        default=[9366, 1928, 9541, 3708, 2847],
        description="Course IDs for each candidate course.",
        json_schema_extra={"example": [9366, 1928, 9541, 3708, 2847]},
    )
    course_names: list[str] = Field(
        default=["Python for Beginners", "Cybersecurity for Professionals", "DevOps and Continuous Deployment", "Project Management Fundamentals", "Advanced Python"],
        description="Course names for each candidate course.",
        json_schema_extra={"example": ["Python for Beginners", "Cybersecurity for Professionals", "DevOps and Continuous Deployment", "Project Management Fundamentals", "Advanced Python"]},
    )
    instructors: list[str] = Field(
        default=["Emma Harris", "Alexander Young", "Dr. Mia Walker", "Benjamin Lewis", "John Smith"],
        description="Instructor names for each candidate course.",
        json_schema_extra={"example": ["Emma Harris", "Alexander Young", "Dr. Mia Walker", "Benjamin Lewis", "John Smith"]},
    )
    
    # Numerical features
    course_duration_hours: list[float] = Field(
        default=[39.1, 36.3, 13.4, 58.3, 45.2],
        description="Course duration in hours for each candidate course.",
        json_schema_extra={"example": [39.1, 36.3, 13.4, 58.3, 45.2]},
    )
    enrollment_numbers: list[int] = Field(
        default=[21600, 15379, 6431, 48245, 32100],
        description="Number of enrollments for each candidate course.",
        json_schema_extra={"example": [21600, 15379, 6431, 48245, 32100]},
    )
    course_price: list[float] = Field(
        default=[317.5, 40.99, 380.81, 342.8, 250.0],
        description="Price of each candidate course.",
        json_schema_extra={"example": [317.5, 40.99, 380.81, 342.8, 250.0]},
    )
    feedback_score: list[float] = Field(
        default=[0.797, 0.77, 0.772, 0.969, 0.85],
        description="Feedback score for each candidate course.",
        json_schema_extra={"example": [0.797, 0.77, 0.772, 0.969, 0.85]},
    )
    time_spent_hours: list[float] = Field(
        default=[17.6, 28.97, 52.44, 22.29, 35.0],
        description="Time spent on each course in hours.",
        json_schema_extra={"example": [17.6, 28.97, 52.44, 22.29, 35.0]},
    )
    previous_courses_taken: list[int] = Field(
        default=[4, 9, 4, 6, 5],
        description="Number of previous courses taken by user.",
        json_schema_extra={"example": [4, 9, 4, 6, 5]},
    )
    
    # Categorical features
    difficulty_level: list[str] = Field(
        default=["Beginner", "Beginner", "Beginner", "Beginner", "Intermediate"],
        description="Difficulty level of each candidate course (e.g., 'Beginner', 'Intermediate', 'Advanced').",
        json_schema_extra={"example": ["Beginner", "Beginner", "Beginner", "Beginner", "Intermediate"]},
    )
    certification_offered: list[str] = Field(
        default=["Yes", "Yes", "Yes", "Yes", "Yes"],
        description="Whether certification is offered ('Yes' or 'No').",
        json_schema_extra={"example": ["Yes", "Yes", "Yes", "Yes", "Yes"]},
    )
    study_material_available: list[str] = Field(
        default=["Yes", "Yes", "Yes", "No", "Yes"],
        description="Whether study material is available ('Yes' or 'No').",
        json_schema_extra={"example": ["Yes", "Yes", "Yes", "No", "Yes"]},
    )


class CollaborativeFilteringRequest(BaseModel):
    """Request for Collaborative Filtering (SVD-based)"""
    user_id: int = Field(default=2005, description="User ID to generate recommendations for. Must exist in training data.", json_schema_extra={"example": 2005})
    top_n: int = Field(default=5, ge=1, le=20, description="Number of recommendations to return.", json_schema_extra={"example": 5})
    model_type: Literal["collaborative_filtering"] = Field(
        default="collaborative_filtering",
        description="Collaborative filtering model using TruncatedSVD.",
    )
    candidate_course_ids: list[int] = Field(
        default=[2, 6703, 6685, 6687, 6689],
        description="List of candidate course IDs to rank and recommend from.",
        json_schema_extra={"example": [2, 6703, 6685, 6687, 6689]},
    )


class RecommendationResponse(BaseModel):
    user_id: int
    top_n: int
    model_type: str
    recommendations: list


class DiscoveryRequest(BaseModel):
    """Request for discovering NEW courses based on user preferences."""
    user_id: int = Field(default=15796, description="User ID", json_schema_extra={"example": 15796})
    top_n: int = Field(default=5, ge=1, le=20, description="Number of recommendations", json_schema_extra={"example": 5})
    model_type: Literal["content_based_random_forest", "collaborative_filtering"] = Field(
        default="collaborative_filtering",
        description="Recommendation algorithm to use",
    )
    courses_already_taken: list[int] = Field(
        default=[9366, 1928],
        description="Course IDs the user has already taken (to exclude from recommendations)",
        json_schema_extra={"example": [9366, 1928]},
    )
    previous_courses_taken: int = Field(
        default=4,
        ge=0,
        description="Number of courses user has completed (for content-based model)",
        json_schema_extra={"example": 4},
    )


class HealthResponse(BaseModel):
    status: str
    service: str
    timestamp: str
    version: str
    environment: str


class MetricsResponse(BaseModel):
    requests_total: int
    requests_success: int
    requests_error: int
    avg_response_time_ms: float


class AppMetrics:
    def __init__(self):
        self.requests_total = 0
        self.requests_success = 0
        self.requests_error = 0
        self.response_times = []

    def record_request(self, success: bool, response_time_ms: float):
        self.requests_total += 1
        if success:
            self.requests_success += 1
        else:
            self.requests_error += 1
        self.response_times.append(response_time_ms)

    def get_metrics(self) -> dict:
        avg_response_time = (
            sum(self.response_times) / len(self.response_times)
            if self.response_times
            else 0.0
        )
        return {
            "requests_total": self.requests_total,
            "requests_success": self.requests_success,
            "requests_error": self.requests_error,
            "avg_response_time_ms": avg_response_time,
        }


metrics = AppMetrics()


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint for orchestration and monitoring."""
    return {
        "status": "ok",
        "service": "course-recommendation-api",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "environment": os.getenv("PROJECT_ENV", "unknown"),
    }


@app.get("/metrics", response_model=MetricsResponse)
def get_metrics():
    """Metrics endpoint for monitoring API performance."""
    metrics_data = metrics.get_metrics()
    return {
        "requests_total": metrics_data["requests_total"],
        "requests_success": metrics_data["requests_success"],
        "requests_error": metrics_data["requests_error"],
        "avg_response_time_ms": metrics_data["avg_response_time_ms"],
    }


@app.post("/recommend/content-based", response_model=RecommendationResponse)
def recommend_content_based(payload: ContentBasedFeaturesRequest):
    
    start_time = time.time()
    try:
        candidate_features = []
        for i in range(len(payload.course_ids)):
            feature_dict = {
                "user_id": payload.user_id,
                "course_id": payload.course_ids[i],
                "course_name": payload.course_names[i],
                "instructor": payload.instructors[i],
                "course_duration_hours": float(payload.course_duration_hours[i]),
                "enrollment_numbers": int(payload.enrollment_numbers[i]),
                "course_price": float(payload.course_price[i]),
                "feedback_score": float(payload.feedback_score[i]),
                "time_spent_hours": float(payload.time_spent_hours[i]),
                "previous_courses_taken": int(payload.previous_courses_taken[i]),
                "difficulty_level": payload.difficulty_level[i],
                "certification_offered": payload.certification_offered[i],
                "study_material_available": payload.study_material_available[i],
            }
            candidate_features.append(feature_dict)
        
        result = predict_recommendations(
            user_id=payload.user_id,
            top_n=payload.top_n,
            model_type="content_based_random_forest",
            candidate_features=candidate_features,
        )
        response_time_ms = (time.time() - start_time) * 1000
        metrics.record_request(success=True, response_time_ms=response_time_ms)
        return result
    except FileNotFoundError as exc:
        response_time_ms = (time.time() - start_time) * 1000
        metrics.record_request(success=False, response_time_ms=response_time_ms)
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        response_time_ms = (time.time() - start_time) * 1000
        metrics.record_request(success=False, response_time_ms=response_time_ms)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        response_time_ms = (time.time() - start_time) * 1000
        metrics.record_request(success=False, response_time_ms=response_time_ms)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(exc)}") from exc


@app.post("/recommend/collaborative", response_model=RecommendationResponse)
def recommend_collaborative(payload: CollaborativeFilteringRequest):
    
    start_time = time.time()
    try:
        result = predict_recommendations(
            user_id=payload.user_id,
            top_n=payload.top_n,
            model_type="collaborative_filtering",
            candidate_course_ids=payload.candidate_course_ids,
        )
        response_time_ms = (time.time() - start_time) * 1000
        metrics.record_request(success=True, response_time_ms=response_time_ms)
        return result
    except FileNotFoundError as exc:
        response_time_ms = (time.time() - start_time) * 1000
        metrics.record_request(success=False, response_time_ms=response_time_ms)
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        response_time_ms = (time.time() - start_time) * 1000
        metrics.record_request(success=False, response_time_ms=response_time_ms)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        response_time_ms = (time.time() - start_time) * 1000
        metrics.record_request(success=False, response_time_ms=response_time_ms)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(exc)}") from exc


@app.post("/discover", response_model=RecommendationResponse)
def discover_new_courses(payload: DiscoveryRequest):
    """Discover new courses for a user using the trained recommendation bundle."""
    start_time = time.time()
    try:
        if payload.model_type == "content_based_random_forest":
            all_courses = []
            course_df = pd.read_csv("data/processed/engineered_recommend_data.csv")
            for _, row in course_df.drop_duplicates(subset=["course_id"]).iterrows():
                if int(row["course_id"]) in payload.courses_already_taken:
                    continue
                all_courses.append({
                    "course_id": int(row["course_id"]),
                    "course_name": row["course_name"],
                    "instructor": row["instructor"],
                    "course_duration_hours": float(row["course_duration_hours"]),
                    "certification_offered": row["certification_offered"],
                    "difficulty_level": row["difficulty_level"],
                    "enrollment_numbers": int(row["enrollment_numbers"]),
                    "course_price": float(row["course_price"]),
                    "feedback_score": float(row["feedback_score"]),
                    "study_material_available": row["study_material_available"],
                    "time_spent_hours": float(row["time_spent_hours"]),
                    "previous_courses_taken": int(payload.previous_courses_taken),
                })
            result = predict_recommendations(
                user_id=payload.user_id,
                top_n=payload.top_n,
                model_type="content_based_random_forest",
                candidate_features=all_courses,
            )
        else:
            result = predict_recommendations(
                user_id=payload.user_id,
                top_n=payload.top_n,
                model_type="collaborative_filtering",
                candidate_course_ids=[course_id for course_id in pd.read_csv("data/processed/engineered_recommend_data.csv")["course_id"].drop_duplicates().tolist() if course_id not in payload.courses_already_taken],
            )

        response_time_ms = (time.time() - start_time) * 1000
        metrics.record_request(success=True, response_time_ms=response_time_ms)
        return result
    except FileNotFoundError as exc:
        response_time_ms = (time.time() - start_time) * 1000
        metrics.record_request(success=False, response_time_ms=response_time_ms)
        raise HTTPException(status_code=404, detail=f"Model or data not found: {str(exc)}") from exc
    except ValueError as exc:
        response_time_ms = (time.time() - start_time) * 1000
        metrics.record_request(success=False, response_time_ms=response_time_ms)
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(exc)}") from exc
    except Exception as exc:
        response_time_ms = (time.time() - start_time) * 1000
        metrics.record_request(success=False, response_time_ms=response_time_ms)
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(exc)}") from exc


@app.post("/recommend", response_model=RecommendationResponse, deprecated=True)
def recommend_legacy(payload: Any):
    """Legacy endpoint. Use /recommend/collaborative or /recommend/content-based instead."""
    raise HTTPException(
        status_code=410,
        detail="This endpoint is deprecated. Use /recommend/collaborative or /recommend/content-based instead.",
    )
