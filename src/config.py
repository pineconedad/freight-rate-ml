from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
OUTPUT_DIR = ROOT / "outputs"
REPORT_DIR = ROOT / "reports"
SCORER_DIR = ROOT / "scorer_results"

TRAIN_PATH = DATA_DIR / "train-test.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
TEMPLATE_PATH = DATA_DIR / "validation-predictions-template.csv"
DECEMBER_PATH = DATA_DIR / "december-chart-inputs.csv"

TARGET = "posted_rate"
RANDOM_SEED = 42
DATE_ORIGIN = "2025-01-01"

TUNING_FOLDS = (
    ("june", "2025-06-01", "2025-06-30"),
    ("july", "2025-07-01", "2025-07-31"),
    ("august", "2025-08-01", "2025-08-31"),
)
LOCKED_HOLDOUT = ("locked_sep_oct", "2025-09-01", "2025-10-31")

STRUCTURAL_CATEGORICAL = ["equipment"]
ROUTE_CATEGORICAL = ["equipment", "pickup", "delivery", "route"]
CHART_CATEGORICAL = ["equipment", "pickup", "delivery", "route"]

STRUCTURAL_NUMERIC = [
    "distance",
    "log_distance",
    "inverse_distance",
    "weight_abs",
    "weight_missing",
    "weight_negative",
    "market_index",
    "market_missing",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "lat_delta",
    "lon_delta",
    "day_index",
    "dow_sin",
    "dow_cos",
    "year_sin",
    "year_cos",
]

CHART_NUMERIC = [
    "distance",
    "log_distance",
    "inverse_distance",
    "weight_abs",
    "weight_missing",
    "weight_negative",
    "day_index",
    "dow_sin",
    "dow_cos",
    "year_sin",
    "year_cos",
]


def ensure_output_directories() -> None:
    for path in (ARTIFACT_DIR, OUTPUT_DIR, REPORT_DIR, SCORER_DIR):
        path.mkdir(parents=True, exist_ok=True)
