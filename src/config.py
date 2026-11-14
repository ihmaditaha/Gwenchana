from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "student-mental-health.csv"
CLEAN_DATA_PATH = BASE_DIR / "data" / "processed" / "df_clean.csv"

TARGET = "Mental_Health_Risk"

MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "best_model_pipeline.joblib"

REPORT_DIR = BASE_DIR / "reports"
METRICS_PATH = REPORT_DIR / "metrics.json"
CONFUSION_MATRIX_PATH = REPORT_DIR / "confusion_matrix.png"


RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_SPLITS = 5

NUMERICAL_FEATURES = [
    "Age",
    "Year_of_Study",
    "Sleep_Hours",
    "Study_Hours_Per_Week",
    "GPA",
    "Social_Media_Hours",
    "Screen_Time_Hours",
    "Financial_Stress",
    "Family_Support",
    "Attendance_Rate",
]

NOMINAL_FEATURES = ["Gender", "Part_Time_Job", "Previous_Mental_Health_History"]

ORDINAL_FEATURES = ["Physical_Activity", "Diet_Quality"]