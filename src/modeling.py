from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import RANDOM_SEED, TARGET
from src.data import make_features


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    mode: str
    target: str
    loss: str = "MAE"
    include_quote: bool = False
    eligible: bool = True


CANDIDATES = (
    Candidate("baseline_equipment_distance", "baseline", "structural", "rpm"),
    Candidate("ridge_rpm", "ridge", "structural", "rpm"),
    Candidate("cat_structural_rpm", "catboost", "structural", "rpm"),
    Candidate("cat_structural_rpm_squared", "catboost", "structural", "rpm", loss="RMSE"),
    Candidate("cat_structural_direct", "catboost", "structural", "direct"),
    Candidate("cat_route_rpm", "catboost", "route", "rpm"),
    Candidate(
        "cat_structural_rpm_with_quote",
        "catboost",
        "structural",
        "rpm",
        include_quote=True,
        eligible=False,
    ),
)


class EquipmentDistanceBaseline:
    def fit(self, frame: pd.DataFrame) -> "EquipmentDistanceBaseline":
        working = frame.copy()
        working["rpm"] = working[TARGET] / working["distance"]
        working["distance_band"] = pd.cut(
            working["distance"],
            bins=[0, 200, 500, 1_000, 1_500, 2_500, np.inf],
            labels=False,
            include_lowest=True,
        )
        self.lookup = working.groupby(["equipment", "distance_band"])["rpm"].median()
        self.equipment_lookup = working.groupby("equipment")["rpm"].median()
        self.global_rpm = float(working["rpm"].median())
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        bands = pd.cut(
            frame["distance"],
            bins=[0, 200, 500, 1_000, 1_500, 2_500, np.inf],
            labels=False,
            include_lowest=True,
        )
        values = []
        for equipment, band in zip(frame["equipment"], bands):
            value = self.lookup.get((equipment, band), np.nan)
            if pd.isna(value):
                value = self.equipment_lookup.get(equipment, self.global_rpm)
            values.append(float(value))
        return np.asarray(values) * frame["distance"].to_numpy()


class RidgeRpmModel:
    def fit(self, frame: pd.DataFrame) -> "RidgeRpmModel":
        features, categorical = make_features(frame, mode="structural")
        numeric = [column for column in features if column not in categorical]
        processor = ColumnTransformer(
            [
                ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
                ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ]
        )
        self.model = make_pipeline(processor, Ridge(alpha=10.0))
        self.model.fit(features, frame[TARGET] / frame["distance"])
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        features, _ = make_features(frame, mode="structural")
        rpm = self.model.predict(features)
        return rpm * frame["distance"].to_numpy()


class FreightCatBoost:
    def __init__(
        self,
        candidate: Candidate,
        *,
        iterations: int = 700,
        depth: int = 6,
        learning_rate: float = 0.05,
    ) -> None:
        self.candidate = candidate
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate

    def fit(self, frame: pd.DataFrame, evaluation: pd.DataFrame | None = None) -> "FreightCatBoost":
        features, categorical = make_features(
            frame,
            mode=self.candidate.mode,
            include_quote=self.candidate.include_quote,
        )
        if self.candidate.target == "rpm":
            target = frame[TARGET] / frame["distance"]
            distance = frame["distance"].to_numpy()
            sample_weight = distance**2 if self.candidate.loss == "RMSE" else distance
        else:
            target = frame[TARGET]
            sample_weight = None

        self.model = CatBoostRegressor(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            loss_function=self.candidate.loss,
            eval_metric="MAE",
            l2_leaf_reg=10.0,
            random_seed=RANDOM_SEED,
            random_strength=0.5,
            allow_writing_files=False,
            verbose=False,
        )
        fit_kwargs: dict[str, object] = {
            "X": features,
            "y": target,
            "cat_features": categorical,
            "sample_weight": sample_weight,
        }
        self.model.fit(**fit_kwargs)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        features, _ = make_features(
            frame,
            mode=self.candidate.mode,
            include_quote=self.candidate.include_quote,
        )
        prediction = np.asarray(self.model.predict(features), dtype=float)
        if self.candidate.target == "rpm":
            prediction *= frame["distance"].to_numpy()
        return prediction

    def save(self, path: str) -> None:
        self.model.save_model(path)


def fit_candidate(candidate: Candidate, fit: pd.DataFrame, evaluation: pd.DataFrame | None = None):
    if candidate.family == "baseline":
        return EquipmentDistanceBaseline().fit(fit)
    if candidate.family == "ridge":
        return RidgeRpmModel().fit(fit)
    return FreightCatBoost(candidate).fit(fit, evaluation)
