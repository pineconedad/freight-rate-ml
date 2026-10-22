from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import ARTIFACT_DIR, LOCKED_HOLDOUT, RANDOM_SEED, TARGET, TUNING_FOLDS, ensure_output_directories
from src.data import raw_input_hashes, read_inputs, split_before_dates
from src.metrics import regression_metrics, sliced_metrics
from src.modeling import CANDIDATES, Candidate, fit_candidate


def _metric_row(candidate: Candidate, fold: str, holdout: pd.DataFrame, prediction: np.ndarray) -> dict[str, object]:
    return {
        "candidate": candidate.name,
        "fold": fold,
        "eligible": candidate.eligible,
        "rows": len(holdout),
        **regression_metrics(holdout[TARGET].to_numpy(), prediction, holdout["distance"].to_numpy()),
    }


def _select_primary(results: pd.DataFrame) -> str:
    primary_candidates = {
        "cat_structural_rpm",
        "cat_structural_rpm_squared",
        "cat_structural_direct",
    }
    eligible = results.loc[results["candidate"].isin(primary_candidates)].copy()
    baseline = results.loc[results["candidate"].eq("baseline_equipment_distance"), ["fold", "mae", "rmse"]]
    baseline = baseline.rename(columns={"mae": "baseline_mae", "rmse": "baseline_rmse"})
    eligible = eligible.merge(baseline, on="fold", how="left")
    eligible["composite"] = 0.5 * eligible["mae"] / eligible["baseline_mae"] + 0.5 * eligible["rmse"] / eligible["baseline_rmse"]
    ranking = eligible.groupby("candidate").agg(composite=("composite", "mean"), worst_mae=("mae", "max"))
    return str(ranking.sort_values(["composite", "worst_mae"]).index[0])


def _blend_predictions(oof: pd.DataFrame, primary: str) -> tuple[float, pd.DataFrame]:
    structural = oof.loc[oof["candidate"].eq(primary)].set_index("row_index")
    route = oof.loc[oof["candidate"].eq("cat_route_rpm")].set_index("row_index")
    common = structural.index.intersection(route.index)
    rows = []
    for structural_weight in np.linspace(0.0, 1.0, 21):
        prediction = (
            structural_weight * structural.loc[common, "prediction"].to_numpy()
            + (1 - structural_weight) * route.loc[common, "prediction"].to_numpy()
        )
        metrics = regression_metrics(
            structural.loc[common, "actual"].to_numpy(),
            prediction,
            structural.loc[common, "distance"].to_numpy(),
        )
        rows.append({"structural_weight": structural_weight, **metrics})
    table = pd.DataFrame(rows)
    table["composite_rank"] = table["mae"].rank(pct=True) + table["rmse"].rank(pct=True)
    best = table.sort_values(["composite_rank", "structural_weight"], ascending=[True, False]).iloc[0]
    best_weight = float(best["structural_weight"])
    # Require a material gain before accepting category complexity.
    pure = table.loc[np.isclose(table["structural_weight"], 1.0)].iloc[0]
    relative_gain = (pure["mae"] - best["mae"]) / pure["mae"]
    if relative_gain < 0.005:
        best_weight = 1.0
    return best_weight, table


def _cold_start_test(train: pd.DataFrame, primary_candidate: Candidate) -> pd.DataFrame:
    before, later = split_before_dates(train, "2025-07-01", "2025-08-31")
    city_counts = pd.concat([later["pickup"], later["delivery"]]).value_counts()
    eligible_cities = city_counts[city_counts >= 80].index.to_numpy()
    rng = np.random.default_rng(RANDOM_SEED)
    rows: list[dict[str, object]] = []
    route_candidate = next(candidate for candidate in CANDIDATES if candidate.name == "cat_route_rpm")
    for repetition in range(3):
        held_cities = set(rng.choice(eligible_cities, size=min(8, len(eligible_cities)), replace=False))
        fit_mask = ~before["pickup"].isin(held_cities) & ~before["delivery"].isin(held_cities)
        test_mask = later["pickup"].isin(held_cities) | later["delivery"].isin(held_cities)
        fit = before.loc[fit_mask]
        holdout = later.loc[test_mask]
        for candidate in (primary_candidate, route_candidate):
            model = fit_candidate(candidate, fit)
            prediction = model.predict(holdout)
            rows.append(
                {
                    "repetition": repetition,
                    "candidate": candidate.name,
                    "held_cities": ", ".join(sorted(held_cities)),
                    "rows": len(holdout),
                    **regression_metrics(
                        holdout[TARGET].to_numpy(), prediction, holdout["distance"].to_numpy()
                    ),
                }
            )
    return pd.DataFrame(rows)


def run_experiments() -> dict[str, object]:
    ensure_output_directories()
    train, _, _, _ = read_inputs()
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []

    for fold_name, start, end in TUNING_FOLDS:
        fit, holdout = split_before_dates(train, start, end)
        for candidate in CANDIDATES:
            model = fit_candidate(candidate, fit, holdout if candidate.family == "catboost" else None)
            prediction = model.predict(holdout)
            metric_rows.append(_metric_row(candidate, fold_name, holdout, prediction))
            prediction_rows.append(
                pd.DataFrame(
                    {
                        "row_index": holdout.index,
                        "fold": fold_name,
                        "candidate": candidate.name,
                        "actual": holdout[TARGET].to_numpy(),
                        "distance": holdout["distance"].to_numpy(),
                        "prediction": prediction,
                    }
                )
            )

    results = pd.DataFrame(metric_rows)
    oof = pd.concat(prediction_rows, ignore_index=True)
    primary_name = _select_primary(results)
    structural_weight, blend_table = _blend_predictions(oof, primary_name)

    candidate_by_name = {candidate.name: candidate for candidate in CANDIDATES}
    primary_candidate = candidate_by_name[primary_name]
    cold_start = _cold_start_test(train, primary_candidate)
    structural_cold_mae = cold_start.loc[cold_start["candidate"].eq(primary_name), "mae"].mean()
    route_cold_mae = cold_start.loc[cold_start["candidate"].eq("cat_route_rpm"), "mae"].mean()
    if route_cold_mae > structural_cold_mae * 1.05:
        structural_weight = max(structural_weight, 0.75)

    locked_name, locked_start, locked_end = LOCKED_HOLDOUT
    locked_fit, locked = split_before_dates(train, locked_start, locked_end)
    primary_model = fit_candidate(candidate_by_name[primary_name], locked_fit, locked)
    primary_prediction = primary_model.predict(locked)
    if structural_weight < 1.0 and primary_name != "cat_route_rpm":
        route_model = fit_candidate(candidate_by_name["cat_route_rpm"], locked_fit, locked)
        route_prediction = route_model.predict(locked)
        locked_prediction = structural_weight * primary_prediction + (1 - structural_weight) * route_prediction
    else:
        locked_prediction = primary_prediction

    locked_metrics = regression_metrics(
        locked[TARGET].to_numpy(), locked_prediction, locked["distance"].to_numpy()
    )
    locked_output = locked[["load_id", "date", "pickup", "delivery", "equipment", "distance", "weight", "market_index", TARGET]].copy()
    locked_output["prediction"] = locked_prediction
    locked_output.to_csv(ARTIFACT_DIR / "locked_holdout_predictions.csv", index=False)
    sliced_metrics(locked, locked_prediction).to_csv(ARTIFACT_DIR / "locked_holdout_slices.csv", index=False)
    results.to_csv(ARTIFACT_DIR / "tuning_metrics.csv", index=False)
    blend_table.to_csv(ARTIFACT_DIR / "blend_search.csv", index=False)
    cold_start.to_csv(ARTIFACT_DIR / "cold_start_metrics.csv", index=False)

    monthly_quote = pd.read_csv(ARTIFACT_DIR / "monthly_quote_relationship.csv")
    quote_corr = monthly_quote["quote_rpm_correlation"]
    quote_unstable = bool((quote_corr > 0.5).any() and (quote_corr < -0.5).any())
    decision = {
        "primary_candidate": primary_name,
        "structural_weight": structural_weight,
        "route_weight": 1.0 - structural_weight,
        "quote_excluded": quote_unstable,
        "quote_reason": "Monthly target relationship reverses sign" if quote_unstable else "No large sign reversal detected",
        "locked_holdout": {"name": locked_name, **locked_metrics},
        "cold_start": {
            "primary_candidate": primary_name,
            "structural_mean_mae": float(structural_cold_mae),
            "route_mean_mae": float(route_cold_mae),
        },
        "raw_input_sha256": raw_input_hashes(),
    }
    with (ARTIFACT_DIR / "model_decision.json").open("w", encoding="utf-8") as handle:
        json.dump(decision, handle, indent=2)
    return decision
