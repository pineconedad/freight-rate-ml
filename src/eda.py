from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import ARTIFACT_DIR, RANDOM_SEED, TARGET, ensure_output_directories
from src.data import raw_input_hashes, read_inputs


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"cannot serialize {type(value)}")


def _save_target_figure(train: pd.DataFrame) -> None:
    rpm = train[TARGET] / train["distance"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)
    axes[0].hist(train[TARGET], bins=80, color="#145A64", alpha=0.9)
    axes[0].set_title("Posted-rate distribution")
    axes[0].set_xlabel("Posted rate ($)")
    axes[0].set_ylabel("Loads")
    axes[1].hist(rpm.clip(upper=6), bins=80, color="#B75D3E", alpha=0.9)
    axes[1].axvline(0.8, color="#333333", linestyle="--", linewidth=1)
    axes[1].axvline(4.0, color="#333333", linestyle="--", linewidth=1)
    axes[1].set_title("Rate per mile, clipped for display")
    axes[1].set_xlabel("Rate per mile ($)")
    figure.tight_layout()
    figure.savefig(ARTIFACT_DIR / "eda_target_distribution.png", bbox_inches="tight")
    plt.close(figure)


def _save_distance_figure(train: pd.DataFrame) -> None:
    sample = train.sample(min(7_500, len(train)), random_state=RANDOM_SEED).copy()
    sample["rpm"] = sample[TARGET] / sample["distance"]
    sample = sample.loc[sample["rpm"].between(0.8, 4.0)]
    figure, axis = plt.subplots(figsize=(8.5, 4.8), dpi=150)
    axis.scatter(sample["distance"], sample["rpm"], s=6, alpha=0.18, color="#145A64")
    bins = pd.qcut(train["distance"], 20, duplicates="drop")
    trend = train.assign(rpm=train[TARGET] / train["distance"]).groupby(bins, observed=True).agg(
        distance=("distance", "median"), rpm=("rpm", "median")
    )
    axis.plot(trend["distance"], trend["rpm"], color="#B75D3E", linewidth=2.5, label="Bin median")
    axis.set_title("Short-haul loads have a higher rate per mile")
    axis.set_xlabel("Distance (miles)")
    axis.set_ylabel("Rate per mile ($)")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(ARTIFACT_DIR / "eda_distance_rpm.png", bbox_inches="tight")
    plt.close(figure)


def _monthly_quote_table(train: pd.DataFrame) -> pd.DataFrame:
    working = train.copy()
    working["month"] = working["date"].dt.month
    working["rpm"] = working[TARGET] / working["distance"]
    working = working.loc[working["rpm"].between(0.8, 4.0)]
    return (
        working.groupby("month")
        .apply(
            lambda group: pd.Series(
                {
                    "rows": len(group),
                    "quote_rpm_correlation": group["quote_signal"].corr(group["rpm"]),
                    "mean_rpm": group["rpm"].mean(),
                    "mean_market_index": group["market_index"].mean(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )


def _save_temporal_figure(train: pd.DataFrame, monthly_quote: pd.DataFrame) -> None:
    daily = train.assign(rpm=train[TARGET] / train["distance"]).groupby("date").agg(
        median_rpm=("rpm", "median"), market_index=("market_index", "median")
    )
    figure, axes = plt.subplots(2, 1, figsize=(10.5, 7), dpi=150)
    axes[0].plot(daily.index, daily["median_rpm"], color="#145A64", label="Median RPM")
    twin = axes[0].twinx()
    twin.plot(daily.index, daily["market_index"], color="#B75D3E", alpha=0.65, label="Market index")
    axes[0].set_title("Daily pricing and market conditions")
    axes[0].set_ylabel("Median rate per mile")
    twin.set_ylabel("Median market index")
    colors = np.where(monthly_quote["quote_rpm_correlation"] >= 0, "#145A64", "#B75D3E")
    axes[1].bar(monthly_quote["month"], monthly_quote["quote_rpm_correlation"], color=colors)
    axes[1].axhline(0, color="#333333", linewidth=0.8)
    axes[1].set_ylim(-1.05, 1.05)
    axes[1].set_title("Quote signal changes relationship with the target by month")
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("Correlation with rate per mile")
    figure.tight_layout()
    figure.savefig(ARTIFACT_DIR / "eda_temporal_and_quote.png", bbox_inches="tight")
    plt.close(figure)


def run_eda() -> dict[str, object]:
    ensure_output_directories()
    train, validation, template, december = read_inputs()
    # September and October remain target-blind until the locked evaluation.
    discovery = train.loc[train["date"] < pd.Timestamp("2025-09-01")].copy()
    monthly_quote = _monthly_quote_table(discovery)
    monthly_quote.to_csv(ARTIFACT_DIR / "monthly_quote_relationship.csv", index=False)

    train_cities = set(train["pickup"]).union(train["delivery"])
    validation_cities = set(validation["pickup"]).union(validation["delivery"])
    unseen_cities = sorted(validation_cities - train_cities)
    unseen_city_rows = validation["pickup"].isin(unseen_cities) | validation["delivery"].isin(unseen_cities)
    train_routes = set(zip(train["pickup"], train["delivery"]))
    unseen_route_rows = np.fromiter(
        ((pickup, delivery) not in train_routes for pickup, delivery in zip(validation["pickup"], validation["delivery"])),
        dtype=bool,
        count=len(validation),
    )
    rpm = discovery[TARGET] / discovery["distance"]
    quote_corr = monthly_quote["quote_rpm_correlation"]
    quote_unstable = bool((quote_corr > 0.5).any() and (quote_corr < -0.5).any())

    summary: dict[str, object] = {
        "raw_input_sha256": raw_input_hashes(),
        "train": {
            "rows": len(train),
            "date_min": train["date"].min(),
            "date_max": train["date"].max(),
            "missing_weight": int(train["weight"].isna().sum()),
            "negative_weight": int(train["weight"].lt(0).sum()),
            "missing_market_index": int(train["market_index"].isna().sum()),
            "target_analysis_date_max": discovery["date"].max(),
            "target_analysis_rows": len(discovery),
            "target_mean": float(discovery[TARGET].mean()),
            "target_median": float(discovery[TARGET].median()),
            "target_max": float(discovery[TARGET].max()),
            "distance_target_pearson": float(discovery["distance"].corr(discovery[TARGET])),
            "distance_target_spearman": float(discovery["distance"].corr(discovery[TARGET], method="spearman")),
            "rpm_below_0_8": int((rpm < 0.8).sum()),
            "rpm_above_4_0": int((rpm > 4.0).sum()),
        },
        "validation": {
            "rows": len(validation),
            "date_min": validation["date"].min(),
            "date_max": validation["date"].max(),
            "missing_weight": int(validation["weight"].isna().sum()),
            "negative_weight": int(validation["weight"].lt(0).sum()),
            "missing_market_index": int(validation["market_index"].isna().sum()),
            "unseen_cities": unseen_cities,
            "unseen_city_rows": int(unseen_city_rows.sum()),
            "unseen_route_rows": int(unseen_route_rows.sum()),
        },
        "contracts": {
            "template_rows": len(template),
            "december_rows": len(december),
        },
        "discovery_rules": {
            "quote_has_large_sign_reversal": quote_unstable,
            "exclude_quote_from_primary_model": quote_unstable,
            "use_temporal_validation": bool(train["date"].max() < validation["date"].min()),
            "require_cold_start_test": bool(unseen_city_rows.any()),
            "use_robust_training": bool(((rpm < 0.8) | (rpm > 4.0)).mean() > 0.005),
        },
    }
    with (ARTIFACT_DIR / "eda_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=_json_default)

    _save_target_figure(discovery)
    _save_distance_figure(discovery)
    _save_temporal_figure(discovery, monthly_quote)
    return summary
