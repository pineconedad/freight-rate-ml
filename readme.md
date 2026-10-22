# Freight rate prediction assessment

This repository contains a reproducible solution to Spotter's freight-rate prediction exercise. It keeps the supplied CSV files unchanged and records the evidence used to choose the final model.

## Method

The workflow follows the order used in a real model-development project:

1. Validate schemas and audit data quality.
2. Use EDA to turn observations into testable modeling questions.
3. Compare baselines and candidate models on forward-only monthly folds.
4. Stress-test models on cities removed entirely from training.
5. Evaluate the chosen design once on a locked September-October holdout.
6. Refit on all labeled rows and generate final predictions.
7. Validate outputs with the supplied `score.py`.
8. Build the submission report from saved experiment results.

The main model predicts rate per mile with robust CatBoost MAE training. Distance sample weights align RPM error with dollar MAE. The model excludes `load_id` and `quote_signal`. The latter reverses its relationship with the target across historical months and fails forward validation.

## Environment

Python 3.11 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Reproduce everything

```bash
python run_pipeline.py all
```

The full command runs EDA, temporal experiments, cold-start tests, final training, prediction, the supplied scorer, and report generation. Individual stages are also available:

```bash
python run_pipeline.py eda
python run_pipeline.py experiment
python run_pipeline.py train
python run_pipeline.py score
python run_pipeline.py report
```

## Outputs

- `outputs/validation_predictions.csv`: 12,000 final predictions in template order
- `outputs/december_predictions.csv`: 31 fixed-route December predictions
- `scorer_results/candidate_december.png`: chart created by the supplied scorer
- `reports/freight_rate_assessment_report.pdf`: submission report
- `artifacts/model_decision.json`: selected model and locked-holdout metrics
- `artifacts/tuning_metrics.csv`: forward-fold experiment results
- `artifacts/cold_start_metrics.csv`: held-out-city stress tests
- `artifacts/run_manifest.json`: prediction ranges and raw-file hashes

## Validation design

Tuning uses expanding windows ending in June, July, and August. September and October form a locked two-month holdout that matches the final prediction horizon. All splits use complete dates. Random row splitting is intentionally avoided because it would mix future conditions and repeated routes into training.

## December policy

The December chart input omits coordinates, `market_index`, and `quote_signal`. A separate reduced-feature model is trained only on fields available in that file. This avoids inventing missing market values or deriving them from the hidden validation batch.

## Supplied files

The original instructions are in `freight-rate-ml-assessment-1-.pdf`. The supplied scorer is unchanged at `score.py`.
