from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


class PlattCalibrator:
    """Monotonic probability calibration that preserves score ranking."""

    def __init__(self) -> None:
        self.model = LogisticRegression(C=1_000.0, solver="lbfgs")

    def fit(self, probabilities: np.ndarray, y_true: pd.Series | np.ndarray):
        self.model.fit(np.asarray(probabilities).reshape(-1, 1), y_true)
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(
            np.asarray(probabilities).reshape(-1, 1)
        )[:, 1]


def binary_metrics(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    training_seconds: float | None = None,
    ranking_probabilities: np.ndarray | None = None,
) -> dict[str, float]:
    ranking_scores = (
        probabilities if ranking_probabilities is None else ranking_probabilities
    )
    metrics = {
        "roc_auc": float(roc_auc_score(y_true, ranking_scores)),
        "pr_auc": float(average_precision_score(y_true, ranking_scores)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
    }
    if training_seconds is not None:
        metrics["training_seconds"] = float(training_seconds)
    return metrics


def percentile_thresholds(
    validation_probabilities: np.ndarray,
    low_percentile: float = 0.60,
    high_percentile: float = 0.85,
) -> dict[str, float]:
    if not 0 < low_percentile < high_percentile < 1:
        raise ValueError("Expected 0 < low_percentile < high_percentile < 1")
    low, high = np.quantile(
        np.asarray(validation_probabilities),
        [low_percentile, high_percentile],
    )
    return {
        "low_medium": float(low),
        "medium_high": float(high),
        "low_percentile": low_percentile,
        "high_percentile": high_percentile,
    }


def assign_risk_band(
    probabilities: np.ndarray | pd.Series,
    thresholds: Mapping[str, float],
) -> pd.Categorical:
    values = np.asarray(probabilities)
    return pd.cut(
        values,
        bins=[-np.inf, thresholds["low_medium"], thresholds["medium_high"], np.inf],
        labels=["Low", "Medium", "High"],
        right=False,
        ordered=True,
    )


def risk_band_summary(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    thresholds: Mapping[str, float],
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "target": np.asarray(y_true),
            "probability": np.asarray(probabilities),
            "risk_band": assign_risk_band(probabilities, thresholds),
        }
    )
    summary = (
        frame.groupby("risk_band", observed=True)
        .agg(
            applicants=("target", "size"),
            defaults=("target", "sum"),
            observed_default_rate=("target", "mean"),
            mean_predicted_risk=("probability", "mean"),
        )
        .reset_index()
    )
    summary["population_share"] = summary["applicants"] / len(frame)
    total_defaults = max(int(frame["target"].sum()), 1)
    summary["default_capture"] = summary["defaults"] / total_defaults
    return summary


def validate_monotonic_bands(summary: pd.DataFrame) -> bool:
    ordered = (
        summary.set_index("risk_band")
        .reindex(["Low", "Medium", "High"])["observed_default_rate"]
        .dropna()
    )
    return bool(ordered.is_monotonic_increasing and len(ordered) == 3)
