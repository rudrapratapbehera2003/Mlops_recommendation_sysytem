from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class DataContract:

    COLUMN_MAPPING: Dict[str, str] = field(
        default_factory=lambda: {
            "user_id": "user_id",
            "course_id": "course_id",
            "course_name": "course_name",
            "instructor": "instructor",
            "course_duration_hours": "course_duration_hours",
            "certification_offered": "certification_offered",
            "difficulty_level": "difficulty_level",
            "rating": "rating",
            "enrollment_numbers": "enrollment_numbers",
            "course_price": "course_price",
            "feedback_score": "feedback_score",
            "study_material_available": "study_material_available",
            "time_spent_hours": "time_spent_hours",
            "previous_courses_taken": "previous_courses_taken",
        }
    )

    NUMERICAL_COLUMNS: List[str] = field(
        default_factory=lambda: {
            "user_id",
            "course_id",
            "course_duration_hours",
            "rating",
            "enrollment_numbers",
            "course_price",
            "feedback_score",
            "time_spent_hours",
            "previous_courses_taken",
        }
    )

    CATEGORICAL_COLUMNS: List[str] = field(
        default_factory=lambda: {
            "course_name",
            "instructor",
            "certification_offered",
            "difficulty_level",
            "study_material_available",
        }
    )
