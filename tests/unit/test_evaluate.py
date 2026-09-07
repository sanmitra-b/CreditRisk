import numpy as np

from src.ml.evaluate import (
    assign_risk_band,
    percentile_thresholds,
    risk_band_summary,
    validate_monotonic_bands,
)


def test_percentile_bands_have_expected_order() -> None:
    probabilities = np.linspace(0, 1, 101)
    thresholds = percentile_thresholds(probabilities)
    assert thresholds["low_medium"] < thresholds["medium_high"]
    bands = assign_risk_band(np.array([0.1, 0.7, 0.95]), thresholds)
    assert bands.astype(str).tolist() == ["Low", "Medium", "High"]


def test_monotonic_band_validation() -> None:
    y = np.array([0] * 55 + [1] * 5 + [0] * 15 + [1] * 10 + [0] * 3 + [1] * 12)
    probabilities = np.linspace(0, 1, len(y))
    thresholds = percentile_thresholds(probabilities)
    summary = risk_band_summary(y, probabilities, thresholds)
    assert validate_monotonic_bands(summary)

