from __future__ import annotations

import argparse
import json
import subprocess
import sys

from src.config import ARTIFACT_DIR, OUTPUT_DIR, ROOT, SCORER_DIR
from src.data import raw_input_hashes
from src.eda import run_eda
from src.experiments import run_experiments
from src.reporting import generate_report
from src.train import train_and_predict


def run_scorer() -> None:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "score.py"),
            "--predictions",
            str(OUTPUT_DIR / "validation_predictions.csv"),
            "--december-predictions",
            str(OUTPUT_DIR / "december_predictions.csv"),
        ],
        check=True,
        cwd=ROOT,
    )


def all_exist(*paths) -> bool:
    return all(path.exists() for path in paths)


def decision_is_current() -> bool:
    path = ARTIFACT_DIR / "model_decision.json"
    if not path.exists():
        return False
    try:
        with path.open(encoding="utf-8") as handle:
            decision = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    return decision.get("raw_input_sha256") == raw_input_hashes()


def manifest_is_current() -> bool:
    path = ARTIFACT_DIR / "run_manifest.json"
    if not path.exists():
        return False
    try:
        with path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    return manifest.get("raw_input_sha256") == raw_input_hashes()


def main() -> None:
    parser = argparse.ArgumentParser(description="Freight-rate assessment pipeline")
    parser.add_argument("stage", choices=["eda", "experiment", "train", "score", "report", "all"], nargs="?", default="all")
    args = parser.parse_args()

    if args.stage in {"eda", "all"}:
        run_eda()
    eda_ready = all_exist(
        ARTIFACT_DIR / "eda_summary.json",
        ARTIFACT_DIR / "monthly_quote_relationship.csv",
        ARTIFACT_DIR / "eda_distance_rpm.png",
        ARTIFACT_DIR / "eda_temporal_and_quote.png",
    )
    experiment_ready = decision_is_current() and all_exist(
        ARTIFACT_DIR / "tuning_metrics.csv",
        ARTIFACT_DIR / "blend_search.csv",
        ARTIFACT_DIR / "cold_start_metrics.csv",
    )
    outputs_ready = manifest_is_current() and all_exist(
        OUTPUT_DIR / "validation_predictions.csv",
        OUTPUT_DIR / "december_predictions.csv",
    )
    report_ready = all_exist(
        SCORER_DIR / "candidate_december.png",
        ARTIFACT_DIR / "eda_summary.json",
        ARTIFACT_DIR / "model_decision.json",
        ARTIFACT_DIR / "tuning_metrics.csv",
        ARTIFACT_DIR / "cold_start_metrics.csv",
        ARTIFACT_DIR / "run_manifest.json",
        ARTIFACT_DIR / "eda_distance_rpm.png",
        ARTIFACT_DIR / "eda_temporal_and_quote.png",
    )
    report_ready = report_ready and decision_is_current() and manifest_is_current()

    if args.stage == "experiment" and not eda_ready:
        run_eda()
    if args.stage in {"experiment", "all"}:
        run_experiments()
    if args.stage == "train" and not experiment_ready:
        if not eda_ready:
            run_eda()
        run_experiments()
    if args.stage in {"train", "all"}:
        train_and_predict()
    if args.stage == "score" and not outputs_ready:
        if not experiment_ready:
            if not eda_ready:
                run_eda()
            run_experiments()
        train_and_predict()
    if args.stage in {"score", "all"}:
        run_scorer()
    if args.stage == "report" and not report_ready:
        if not outputs_ready:
            if not experiment_ready:
                if not eda_ready:
                    run_eda()
                run_experiments()
            train_and_predict()
        run_scorer()
    if args.stage in {"report", "all"}:
        path = generate_report()
        print(f"Created report: {path}")


if __name__ == "__main__":
    main()
