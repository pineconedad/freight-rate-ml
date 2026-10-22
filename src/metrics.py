from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, median_absolute_error, root_mean_squared_error


def regression_metrics(y_true: np.ndarray, prediction: np.ndarray, distance: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    distance = np.asarray(distance, dtype=float)
    rpm = y_true / distance
    clean = (rpm >= 0.8) & (rpm <= 4.0)
    absolute_error = np.abs(y_true - prediction)
    result = {
        "mae": float(mean_absolute_error(y_true, prediction)),
        "rmse": float(root_mean_squared_error(y_true, prediction)),
        "median_ae": float(median_absolute_error(y_true, prediction)),
        "bias": float(np.mean(prediction - y_true)),
        "p95_ae": float(np.quantile(absolute_error, 0.95)),
        "clean_fraction": float(clean.mean()),
    }
    if clean.any():
        result["clean_mae"] = float(mean_absolute_error(y_true[clean], prediction[clean]))
        result["clean_rmse"] = float(root_mean_squared_error(y_true[clean], prediction[clean]))
    else:
        result["clean_mae"] = float("nan")
        result["clean_rmse"] = float("nan")
    return result


def sliced_metrics(frame: pd.DataFrame, prediction: np.ndarray) -> pd.DataFrame:
    working = frame.copy()
    working["prediction"] = prediction
    working["distance_band"] = pd.qcut(working["distance"], q=4, duplicates="drop")
    working["weight_quality"] = np.select(
        [working["weight"].isna(), working["weight"].lt(0)],
        ["missing", "negative"],
        default="valid",
    )
    working["market_quality"] = np.where(working["market_index"].isna(), "missing", "present")
    rows: list[dict[str, object]] = []
    for column in ("equipment", "distance_band", "weight_quality", "market_quality"):
        for value, group in working.groupby(column, observed=True):
            values = regression_metrics(
                group["posted_rate"].to_numpy(),
                group["prediction"].to_numpy(),
                group["distance"].to_numpy(),
            )
            rows.append({"slice": column, "value": str(value), "rows": len(group), **values})
    return pd.DataFrame(rows)
