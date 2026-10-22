from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from src.config import ARTIFACT_DIR, REPORT_DIR, SCORER_DIR, ensure_output_directories


CANDIDATE_LABELS = {
    "baseline_equipment_distance": "Freight-rule baseline",
    "ridge_rpm": "Ridge RPM",
    "cat_structural_rpm": "Structural CatBoost RPM",
    "cat_structural_rpm_squared": "Structural RPM squared loss",
    "cat_structural_direct": "Structural CatBoost direct",
    "cat_route_rpm": "Route CatBoost RPM",
    "cat_structural_rpm_with_quote": "Structural RPM + quote",
}


def _page_title(figure: plt.Figure, title: str, subtitle: str | None = None) -> None:
    figure.text(0.07, 0.94, title, fontsize=20, fontweight="bold", color="#123F46")
    if subtitle:
        figure.text(0.07, 0.90, subtitle, fontsize=10.5, color="#4E6266")


def _add_wrapped_text(figure: plt.Figure, text: str, *, top: float = 0.85, width: int = 105) -> None:
    wrapped = "\n".join(textwrap.fill(paragraph, width=width) for paragraph in text.split("\n"))
    figure.text(
        0.07,
        top,
        wrapped,
        fontsize=10.2,
        va="top",
        linespacing=1.45,
        color="#24383C",
        parse_math=False,
    )


def _text_page(pdf: PdfPages, title: str, text: str, subtitle: str | None = None) -> None:
    figure = plt.figure(figsize=(8.27, 11.69))
    figure.patch.set_facecolor("white")
    _page_title(figure, title, subtitle)
    _add_wrapped_text(figure, text)
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def _image_page(pdf: PdfPages, title: str, image_path: Path, note: str) -> None:
    figure = plt.figure(figsize=(8.27, 11.69))
    _page_title(figure, title)
    axis = figure.add_axes([0.07, 0.25, 0.86, 0.60])
    axis.imshow(mpimg.imread(image_path))
    axis.axis("off")
    figure.text(
        0.07,
        0.18,
        textwrap.fill(note, 105),
        fontsize=10.2,
        va="top",
        linespacing=1.4,
        parse_math=False,
    )
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def _table_page(pdf: PdfPages, title: str, frame: pd.DataFrame, columns: list[str], note: str) -> None:
    display = frame.loc[:, columns].copy()
    for column in display.select_dtypes(include="number"):
        display[column] = display[column].map(lambda value: f"{value:,.2f}")
    figure, axis = plt.subplots(figsize=(8.27, 11.69))
    axis.axis("off")
    _page_title(figure, title)
    remaining_width = 0.64 / max(1, len(columns) - 1)
    table = axis.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="center",
        colLoc="center",
        colWidths=[0.32] + [remaining_width] * (len(columns) - 1),
        bbox=[0.02, 0.30, 0.96, 0.50],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#145A64")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#EDF3F4")
    figure.text(
        0.07,
        0.22,
        textwrap.fill(note, 105),
        fontsize=10.2,
        va="top",
        linespacing=1.4,
        parse_math=False,
    )
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def generate_report() -> Path:
    ensure_output_directories()
    with (ARTIFACT_DIR / "eda_summary.json").open(encoding="utf-8") as handle:
        eda = json.load(handle)
    with (ARTIFACT_DIR / "model_decision.json").open(encoding="utf-8") as handle:
        decision = json.load(handle)
    with (ARTIFACT_DIR / "run_manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    tuning = pd.read_csv(ARTIFACT_DIR / "tuning_metrics.csv")
    cold = pd.read_csv(ARTIFACT_DIR / "cold_start_metrics.csv")
    report_path = REPORT_DIR / "freight_rate_assessment_report.pdf"

    with PdfPages(report_path) as pdf:
        _text_page(
            pdf,
            "Freight rate prediction assessment",
            "This report follows the order in which the solution was developed: establish the deployment boundary, audit the data, form testable hypotheses, run forward experiments, evaluate once on a locked holdout, and only then train on all labeled data. This order prevents a persuasive explanation from being invented after seeing a preferred result.\n\nThe final task is future-period regression. The labeled data ends on October 31, while every hidden evaluation row falls in November or December. Consequently, model selection used complete-date forward splits rather than random cross-validation.",
            "Reproducible model-development and validation record",
        )
        _text_page(
            pdf,
            "1. Audit findings and resulting questions",
            f"The schema audit found {eda['train']['rows']:,} labeled rows and {eda['validation']['rows']:,} final rows. Target-aware discovery used the {eda['train']['target_analysis_rows']:,} rows through August 31, preserving September and October as a locked holdout. In that discovery period, distance had Pearson correlation {eda['train']['distance_target_pearson']:.3f} with posted rate. There were {eda['train']['negative_weight']} negative training weights, {eda['train']['missing_weight']} missing weights, and {eda['train']['missing_market_index']} missing market indices. The final data introduced {len(eda['validation']['unseen_cities'])} cities absent from training, affecting {eda['validation']['unseen_city_rows']:,} rows.\n\nThese observations generated four experiments rather than immediate conclusions. First, compare raw-rate and rate-per-mile targets. Second, test robust training because the target has isolated extreme RPM values. Third, compare coordinate-first and route-category models under held-out-city tests. Fourth, measure quote-signal stability month by month before deciding whether it is safe for future prediction.",
        )
        _image_page(
            pdf,
            "2. Target structure",
            ARTIFACT_DIR / "eda_distance_rpm.png",
            "The median rate per mile falls with distance, consistent with fixed pickup and delivery costs being spread over more miles. This motivated rate per mile as a candidate target, but the choice was retained only if it beat direct-rate modeling on forward folds.",
        )
        _image_page(
            pdf,
            "3. Temporal stability and leakage audit",
            ARTIFACT_DIR / "eda_temporal_and_quote.png",
            "The quote signal changes from strongly positive to strongly negative across months. Because final predictions occur in unseen future months, this sign reversal is a temporal stability failure. The quote feature was kept as an ablation candidate but excluded from the eligible final models.",
        )
        summary_table = (
            tuning.groupby("candidate", as_index=False)
            .agg(mae=("mae", "mean"), rmse=("rmse", "mean"), clean_mae=("clean_mae", "mean"), bias=("bias", "mean"))
            .sort_values("mae")
        )
        summary_table["candidate"] = summary_table["candidate"].map(CANDIDATE_LABELS).fillna(summary_table["candidate"])
        _table_page(
            pdf,
            "4. Forward-fold model comparison",
            summary_table,
            ["candidate", "mae", "rmse", "clean_mae", "bias"],
            "Metrics are averages across June, July, and August expanding-window folds. MAE describes the ordinary dollar miss. RMSE is reported because the hidden metric is undisclosed, though it is dominated by apparently random target multipliers. The final candidate was selected using normalized MAE and RMSE against the same baseline, with worst-fold error as a tie-breaker.",
        )
        cold_table = cold.groupby("candidate", as_index=False).agg(mae=("mae", "mean"), rmse=("rmse", "mean"), clean_mae=("clean_mae", "mean"))
        cold_table["candidate"] = cold_table["candidate"].map(CANDIDATE_LABELS).fillna(cold_table["candidate"])
        _table_page(
            pdf,
            "5. Cold-start stress test",
            cold_table,
            ["candidate", "mae", "rmse", "clean_mae"],
            "Three groups of eight cities were removed entirely from pre-July fitting data and evaluated on July and August rows. This leaves the September-October locked holdout untouched. Coordinates provide a prediction path for unseen cities. Category-heavy model weight was capped when it degraded held-out-city MAE by more than five percent.",
        )
        locked = decision["locked_holdout"]
        _text_page(
            pdf,
            "6. Locked holdout and final model",
            f"After tuning, the selected primary candidate was {CANDIDATE_LABELS.get(decision['primary_candidate'], decision['primary_candidate'])}. The final blend assigned {decision['structural_weight']:.0%} to the selected primary model and {decision['route_weight']:.0%} to the route-category model. No design choices were changed after the September-October holdout was scored.\n\nLocked-holdout MAE: ${locked['mae']:,.2f}\nLocked-holdout RMSE: ${locked['rmse']:,.2f}\nLocked-holdout median absolute error: ${locked['median_ae']:,.2f}\nLocked-holdout bias: {'-' if locked['bias'] < 0 else '+'}${abs(locked['bias']):,.2f}\nLocked-holdout clean-row MAE: ${locked['clean_mae']:,.2f}\n\nThe final models were then refit on every labeled row. Validation predictions ranged from ${manifest['validation_prediction_min']:,.2f} to ${manifest['validation_prediction_max']:,.2f}, with median ${manifest['validation_prediction_median']:,.2f}.",
        )
        _text_page(
            pdf,
            "7. December inference policy",
            "The chart input omits coordinates, market index, and quote signal. Filling those fields from the hidden validation batch would make the chart transductive and would not represent a row-level deployment path. A separate reduced-feature CatBoost model was therefore trained using only pickup, delivery, distance, equipment, repaired weight, and date features. It uses the same robust rate-per-mile formulation.\n\nThe source December CSV remained unchanged. Predictions were written to outputs/december_predictions.csv, validated by the supplied scorer, and plotted below.",
        )
        _image_page(
            pdf,
            "8. Fixed December prediction chart",
            SCORER_DIR / "candidate_december.png",
            f"The fixed-route predictions range from ${manifest['december_prediction_min']:,.2f} to ${manifest['december_prediction_max']:,.2f}. Only date changes across these 31 rows.",
        )
        _text_page(
            pdf,
            "9. Limitations",
            "The final evaluation metric is not disclosed, so the selection score balances MAE and RMSE relative to a common baseline. Extreme target multipliers appear unpredictable from the supplied features and continue to dominate RMSE. The final period also contains unseen cities and a lower market-index distribution. Forward and cold-start validation measure these risks but cannot remove them.\n\nThe December model intentionally excludes market conditions because they are absent from the chart schema. Its predictions should not be interpreted as market-aware daily quotes. In a production system, feature availability and timing would be formalized before retraining.",
        )
    return report_path
