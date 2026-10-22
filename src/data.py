from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    CHART_CATEGORICAL,
    CHART_NUMERIC,
    DATE_ORIGIN,
    DECEMBER_PATH,
    ROUTE_CATEGORICAL,
    STRUCTURAL_CATEGORICAL,
    STRUCTURAL_NUMERIC,
    TARGET,
    TEMPLATE_PATH,
    TRAIN_PATH,
    VALIDATION_PATH,
)


REQUIRED_BASE_COLUMNS = {
    "load_id",
    "pickup",
    "delivery",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "equipment",
    "weight",
    "date",
    "market_index",
    "quote_signal",
}


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(TRAIN_PATH)
    validation = pd.read_csv(VALIDATION_PATH)
    template = pd.read_csv(TEMPLATE_PATH)
    december = pd.read_csv(DECEMBER_PATH)
    validate_input_schema(train, validation, template, december)
    train["date"] = pd.to_datetime(train["date"], errors="raise")
    validation["date"] = pd.to_datetime(validation["date"], errors="raise")
    december["date"] = pd.to_datetime(december["date"], errors="raise")
    return train, validation, template, december


def validate_input_schema(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    template: pd.DataFrame,
    december: pd.DataFrame,
) -> None:
    missing_train = REQUIRED_BASE_COLUMNS.union({TARGET}) - set(train.columns)
    missing_validation = REQUIRED_BASE_COLUMNS - set(validation.columns)
    if missing_train:
        raise ValueError(f"training data is missing columns: {sorted(missing_train)}")
    if missing_validation:
        raise ValueError(f"validation data is missing columns: {sorted(missing_validation)}")
    if list(template.columns) != ["load_id", "predicted_rate"]:
        raise ValueError("prediction template must contain load_id,predicted_rate")
    expected_december = ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]
    if list(december.columns) != expected_december:
        raise ValueError("December input schema does not match the scorer contract")
    for label, frame in (("train", train), ("validation", validation), ("template", template)):
        if frame["load_id"].isna().any() or frame["load_id"].duplicated().any():
            raise ValueError(f"{label} contains missing or duplicate load_id values")
    if set(validation["load_id"].astype(str)) != set(template["load_id"].astype(str)):
        raise ValueError("validation and template IDs do not match")
    if train[TARGET].isna().any() or not np.isfinite(train[TARGET]).all():
        raise ValueError("training target contains invalid values")
    for label, frame in (("train", train), ("validation", validation)):
        distance = pd.to_numeric(frame["distance"], errors="coerce")
        if distance.isna().any() or not np.isfinite(distance).all() or (distance <= 0).any():
            raise ValueError(f"{label} distance must be finite and positive")
        for column in ("pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "quote_signal"):
            values = pd.to_numeric(frame[column], errors="coerce")
            if values.isna().any() or not np.isfinite(values).all():
                raise ValueError(f"{label} {column} must be finite")
        for column in ("weight", "market_index"):
            values = pd.to_numeric(frame[column], errors="coerce")
            invalid = frame[column].notna() & values.isna()
            if invalid.any() or not np.isfinite(values.dropna()).all():
                raise ValueError(f"{label} {column} contains invalid nonmissing values")
    for column in ("distance", "weight"):
        values = pd.to_numeric(december[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values).all():
            raise ValueError(f"December {column} must be finite")


def make_features(
    frame: pd.DataFrame,
    *,
    mode: str,
    include_quote: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    parsed_distance = pd.to_numeric(result["distance"], errors="coerce")
    if parsed_distance.isna().any() or not np.isfinite(parsed_distance).all() or (parsed_distance <= 0).any():
        raise ValueError("distance must be positive")

    distance = result["distance"].astype(float)
    result["log_distance"] = np.log1p(distance)
    result["inverse_distance"] = 1.0 / distance
    result["weight_missing"] = result["weight"].isna().astype(float)
    result["weight_negative"] = result["weight"].lt(0).fillna(False).astype(float)
    result["weight_abs"] = result["weight"].abs()
    result["market_missing"] = (
        result["market_index"].isna().astype(float) if "market_index" in result else 1.0
    )
    result["route"] = result["pickup"].astype(str) + "|" + result["delivery"].astype(str)

    day_index = (result["date"] - pd.Timestamp(DATE_ORIGIN)).dt.days.astype(float)
    day_of_week = result["date"].dt.dayofweek.astype(float)
    day_of_year = result["date"].dt.dayofyear.astype(float)
    result["day_index"] = day_index
    result["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    result["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7)
    result["year_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    result["year_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)

    if mode == "chart":
        numeric = CHART_NUMERIC.copy()
        categorical = CHART_CATEGORICAL.copy()
    elif mode in {"structural", "route"}:
        result["lat_delta"] = result["delivery_lat"] - result["pickup_lat"]
        result["lon_delta"] = result["delivery_lon"] - result["pickup_lon"]
        numeric = STRUCTURAL_NUMERIC.copy()
        categorical = STRUCTURAL_CATEGORICAL.copy() if mode == "structural" else ROUTE_CATEGORICAL.copy()
        if include_quote:
            numeric.append("quote_signal")
    else:
        raise ValueError(f"unknown feature mode: {mode}")

    for column in categorical:
        result[column] = result[column].fillna("__MISSING__").astype(str)
    features = numeric + categorical
    return result.loc[:, features], categorical


def split_before_dates(
    frame: pd.DataFrame,
    start: str,
    end: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    start_date = pd.Timestamp(start)
    end_date = pd.Timestamp(end)
    fit = frame.loc[frame["date"] < start_date].copy()
    holdout = frame.loc[frame["date"].between(start_date, end_date)].copy()
    if fit.empty or holdout.empty:
        raise ValueError(f"empty temporal split for {start} through {end}")
    if fit["date"].max() >= holdout["date"].min():
        raise AssertionError("temporal split leaks future dates into training")
    return fit, holdout


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_input_hashes() -> dict[str, str]:
    return {
        path.name: file_sha256(path)
        for path in (TRAIN_PATH, VALIDATION_PATH, TEMPLATE_PATH, DECEMBER_PATH)
    }
