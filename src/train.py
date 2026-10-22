from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import ARTIFACT_DIR, OUTPUT_DIR, TARGET, ensure_output_directories
from src.data import raw_input_hashes, read_inputs
from src.modeling import CANDIDATES, Candidate, FreightCatBoost, fit_candidate


def _validate_predictions(output: pd.DataFrame, expected_ids: pd.Series) -> None:
    if list(output.columns) != ["load_id", "predicted_rate"]:
        raise ValueError("prediction output has an invalid schema")
    if len(output) != len(expected_ids):
        raise ValueError("prediction output has the wrong row count")
    if output["load_id"].duplicated().any() or set(output["load_id"]) != set(expected_ids):
        raise ValueError("prediction IDs do not match the template")
    values = pd.to_numeric(output["predicted_rate"], errors="coerce")
    if values.isna().any() or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("predictions must be finite and positive")


def train_and_predict() -> dict[str, object]:
    ensure_output_directories()
    train, validation, template, december = read_inputs()
    with (ARTIFACT_DIR / "model_decision.json").open(encoding="utf-8") as handle:
        decision = json.load(handle)
    current_hashes = raw_input_hashes()
    if decision.get("raw_input_sha256") != current_hashes:
        raise ValueError("model decision artifacts do not match the current raw inputs; rerun experiments")
    candidate_by_name = {candidate.name: candidate for candidate in CANDIDATES}
    primary_candidate = candidate_by_name[decision["primary_candidate"]]
    structural_weight = float(decision["structural_weight"])

    primary_model = fit_candidate(primary_candidate, train)
    primary_prediction = primary_model.predict(validation)
    primary_model.save(str(ARTIFACT_DIR / "primary_model.cbm"))
    if structural_weight < 1.0 and primary_candidate.name != "cat_route_rpm":
        route_candidate = candidate_by_name["cat_route_rpm"]
        route_model = fit_candidate(route_candidate, train)
        route_prediction = route_model.predict(validation)
        route_model.save(str(ARTIFACT_DIR / "route_model.cbm"))
        final_prediction = structural_weight * primary_prediction + (1 - structural_weight) * route_prediction
    else:
        final_prediction = primary_prediction

    if not np.isfinite(final_prediction).all():
        raise ValueError("model produced non-finite validation predictions")
    non_positive = int((final_prediction <= 0).sum())
    final_prediction = np.maximum(final_prediction, 1.0)
    prediction_map = pd.Series(final_prediction, index=validation["load_id"].astype(str))
    output = template[["load_id"]].copy()
    output["predicted_rate"] = output["load_id"].astype(str).map(prediction_map)
    _validate_predictions(output, template["load_id"])
    output.to_csv(OUTPUT_DIR / "validation_predictions.csv", index=False, float_format="%.2f")

    chart_candidate = Candidate("cat_chart_rpm", "catboost", "chart", "rpm")
    chart_model = FreightCatBoost(chart_candidate, iterations=700).fit(train)
    december_prediction = chart_model.predict(december)
    if not np.isfinite(december_prediction).all():
        raise ValueError("model produced non-finite December predictions")
    december_output = december.copy()
    december_output["predicted_rate"] = np.maximum(december_prediction, 1.0)
    december_output["date"] = december_output["date"].dt.strftime("%Y-%m-%d")
    december_output.to_csv(OUTPUT_DIR / "december_predictions.csv", index=False, float_format="%.2f")
    chart_model.save(str(ARTIFACT_DIR / "december_model.cbm"))

    manifest = {
        "primary_candidate": primary_candidate.name,
        "structural_weight": structural_weight,
        "validation_rows": len(output),
        "validation_non_positive_before_floor": non_positive,
        "validation_prediction_min": float(final_prediction.min()),
        "validation_prediction_median": float(np.median(final_prediction)),
        "validation_prediction_max": float(final_prediction.max()),
        "december_prediction_min": float(december_prediction.min()),
        "december_prediction_median": float(np.median(december_prediction)),
        "december_prediction_max": float(december_prediction.max()),
        "raw_input_sha256": current_hashes,
    }
    with (ARTIFACT_DIR / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest
