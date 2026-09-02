import os
from typing import Any, Dict, List

import joblib
import pandas as pd


def load_model(model_path: str):
    """Load the persisted recommendation bundle manifest."""
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    bundle = joblib.load(model_path)

    if isinstance(bundle, dict) and "models" in bundle:
        return bundle

    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError("Loaded model artifact is not a valid recommendation bundle.")
    return bundle


def load_all_courses(
    data_path: str = "data/processed/engineered_recommend_data.csv",
) -> pd.DataFrame:
    """Load all available courses from dataset."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Course data not found: {data_path}")

    df = pd.read_csv(data_path)
    # Get unique courses
    unique_courses = df.drop_duplicates(subset=["course_id"]).copy()
    return unique_courses


def _recommend_for_cbf(
    bundle: Dict[str, Any],
    user_id: int,
    top_n: int,
    user_preferences: Dict[str, Any],
    exclude_course_ids: List[int] | None = None,
) -> List[Dict[str, Any]]:

    selected_model = (
        bundle.get("models", {}).get("cbf")
        if isinstance(bundle.get("models"), dict)
        else bundle
    )
    model = selected_model.get("model") if isinstance(selected_model, dict) else None
    preprocessor = (
        selected_model.get("preprocessor") if isinstance(selected_model, dict) else None
    )
    feature_cols = (
        selected_model.get("feature_cols", [])
        if isinstance(selected_model, dict)
        else []
    )
    recommendations: List[Dict[str, Any]] = []

    if not hasattr(model, "predict"):
        return recommendations
    if preprocessor is None or not feature_cols:
        raise ValueError(
            "Content-based model bundle is missing the trained preprocessor and feature columns."
        )

    # Load all courses
    all_courses = load_all_courses()
    exclude_ids = set(exclude_course_ids or [])

    # Filter out already taken courses
    available_courses = all_courses[~all_courses["course_id"].isin(exclude_ids)].copy()

    if len(available_courses) == 0:
        return recommendations

    candidate_data = []
    for _, course in available_courses.iterrows():
        feature_dict = {
            "user_id": user_id,
            "course_id": int(course["course_id"]),
            "course_name": course["course_name"],
            "instructor": course["instructor"],
            "course_duration_hours": float(course["course_duration_hours"]),
            "enrollment_numbers": int(course["enrollment_numbers"]),
            "course_price": float(course["course_price"]),
            "feedback_score": float(course["feedback_score"]),
            "study_material_available": course["study_material_available"],
            "time_spent_hours": float(course.get("time_spent_hours", 0)),
            "previous_courses_taken": user_preferences.get(
                "previous_courses_taken", int(course.get("previous_courses_taken", 0))
            ),
            "difficulty_level": course["difficulty_level"],
            "certification_offered": course["certification_offered"],
        }
        candidate_data.append(feature_dict)

    # Convert to DataFrame and prepare features
    raw_df = pd.DataFrame(candidate_data)
    for col in feature_cols:
        if col not in raw_df.columns:
            raw_df[col] = 0
    raw_df = raw_df[feature_cols]

    # Get predictions
    transformed = preprocessor.transform(raw_df)
    scores = model.predict(transformed)

    # Rank by scores
    ranked = sorted(zip(candidate_data, scores), key=lambda x: x[1], reverse=True)

    # Return top_n
    for rank, (candidate, score) in enumerate(ranked[:top_n], start=1):
        course_id = candidate.get("course_id")
        if course_id is None:
            continue
        recommendations.append(
            {
                "rank": rank,
                "user_id": user_id,
                "course_id": int(course_id),
                "course_name": candidate.get("course_name"),
                "score": float(score),
                "model_type": "content_based_random_forest",
            }
        )

    return recommendations


def _recommend_for_cbf_from_candidates(
    bundle: Dict[str, Any],
    user_id: int,
    top_n: int,
    candidate_features: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    selected_model = (
        bundle.get("models", {}).get("cbf")
        if isinstance(bundle.get("models"), dict)
        else bundle
    )
    model = selected_model.get("model") if isinstance(selected_model, dict) else None
    preprocessor = (
        selected_model.get("preprocessor") if isinstance(selected_model, dict) else None
    )
    feature_cols = (
        selected_model.get("feature_cols", [])
        if isinstance(selected_model, dict)
        else []
    )
    recommendations: List[Dict[str, Any]] = []

    if not candidate_features:
        return recommendations
    if not hasattr(model, "predict"):
        return recommendations
    if preprocessor is None or not feature_cols:
        raise ValueError(
            "Content-based model bundle is missing the trained preprocessor and feature columns."
        )

    raw_df = pd.DataFrame(candidate_features)
    for col in feature_cols:
        if col not in raw_df.columns:
            raw_df[col] = 0
    raw_df = raw_df[feature_cols]
    transformed = preprocessor.transform(raw_df)
    scores = model.predict(transformed)

    ranked = sorted(zip(candidate_features, scores), key=lambda x: x[1], reverse=True)

    for rank, (candidate, score) in enumerate(ranked[:top_n], start=1):
        course_id = candidate.get("course_id")
        if course_id is None:
            continue
        recommendations.append(
            {
                "rank": rank,
                "user_id": user_id,
                "course_id": course_id,
                "score": float(score),
                "model_type": "content_based_random_forest",
            }
        )

    return recommendations


def _recommend_for_cf(
    bundle: Dict[str, Any],
    user_id: int,
    top_n: int,
    exclude_course_ids: List[int] | None = None,
) -> List[Dict[str, Any]]:

    selected_model = (
        bundle.get("models", {}).get("cf")
        if isinstance(bundle.get("models"), dict)
        else bundle
    )
    model_meta = (
        selected_model.get("model", {}) if isinstance(selected_model, dict) else {}
    )
    pred_matrix = (
        model_meta.get("pred_matrix") if isinstance(model_meta, dict) else None
    )

    if pred_matrix is None:
        return []

    # Check if user exists
    if user_id not in pred_matrix.index:
        return []

    # Get all predictions for this user
    user_scores = pred_matrix.loc[user_id]

    # Exclude courses already taken
    exclude_ids = set(exclude_course_ids or [])
    if exclude_ids:
        available_scores = user_scores[~user_scores.index.isin(exclude_ids)]
    else:
        available_scores = user_scores

    # Get top_n
    ranked = available_scores.sort_values(ascending=False).head(top_n)

    return [
        {
            "rank": idx,
            "user_id": user_id,
            "course_id": int(course_id),
            "score": float(score),
            "model_type": "collaborative_filtering",
        }
        for idx, (course_id, score) in enumerate(ranked.items(), start=1)
    ]


def _recommend_for_cf_from_candidates(
    bundle: Dict[str, Any], user_id: int, top_n: int, candidate_course_ids: List[int]
) -> List[Dict[str, Any]]:

    selected_model = (
        bundle.get("models", {}).get("cf")
        if isinstance(bundle.get("models"), dict)
        else bundle
    )
    model_meta = (
        selected_model.get("model", {}) if isinstance(selected_model, dict) else {}
    )
    pred_matrix = (
        model_meta.get("pred_matrix") if isinstance(model_meta, dict) else None
    )

    if pred_matrix is None:
        return []

    if user_id not in pred_matrix.index:
        return []

    user_scores = pred_matrix.loc[user_id]
    filtered_scores = user_scores[user_scores.index.isin(candidate_course_ids)]
    ranked = filtered_scores.sort_values(ascending=False).head(top_n)

    return [
        {
            "rank": idx,
            "user_id": user_id,
            "course_id": int(course_id),
            "score": float(score),
            "model_type": "collaborative_filtering",
        }
        for idx, (course_id, score) in enumerate(ranked.items(), start=1)
    ]


def predict_recommendations(
    user_id: int,
    top_n: int = 10,
    model_path: str = "models/recommender_model.joblib",
    model_type: str | None = None,
    user_preferences: Dict[str, Any] | None = None,
    user_courses_taken: List[int] | None = None,
    candidate_features: List[Dict[str, Any]] | None = None,
    candidate_course_ids: List[int] | None = None,
) -> Dict[str, Any]:

    bundle = load_model(model_path)
    selected_model_type = (
        model_type or bundle.get("selected_model") or "content_based_random_forest"
    )

    if user_preferences is not None and user_courses_taken is not None:
        if selected_model_type == "content_based_random_forest":
            recommendations = _recommend_for_cbf(
                bundle, user_id, top_n, user_preferences, user_courses_taken
            )
        elif selected_model_type == "collaborative_filtering":
            recommendations = _recommend_for_cf(
                bundle, user_id, top_n, user_courses_taken
            )
        else:
            raise ValueError(f"Unsupported model type: {selected_model_type}")

    elif candidate_features is not None or candidate_course_ids is not None:
        if selected_model_type == "content_based_random_forest":
            recommendations = _recommend_for_cbf_from_candidates(
                bundle, user_id, top_n, candidate_features or []
            )
        elif selected_model_type == "collaborative_filtering":
            recommendations = _recommend_for_cf_from_candidates(
                bundle, user_id, top_n, candidate_course_ids or []
            )
        else:
            raise ValueError(f"Unsupported model type: {selected_model_type}")
    else:
        raise ValueError(
            "Must provide either user_preferences+user_courses_taken OR candidate_features/candidate_course_ids"
        )

    return {
        "user_id": user_id,
        "top_n": top_n,
        "model_type": selected_model_type,
        "recommendations": recommendations[:top_n],
    }
