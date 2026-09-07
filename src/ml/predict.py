from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.ml.evaluate import assign_risk_band


class CreditRiskPredictor:
    def __init__(self, bundle_path: str | Path):
        self.bundle_path = Path(bundle_path)
        self.bundle = joblib.load(self.bundle_path)
        required = {"model", "preprocessor", "selected_features", "risk_thresholds"}
        missing = required - set(self.bundle)
        if missing:
            raise ValueError(f"Invalid EBM bundle; missing keys: {sorted(missing)}")

    @property
    def selected_features(self) -> list[str]:
        return list(self.bundle["selected_features"])

    def predict(self, applicants: pd.DataFrame) -> pd.DataFrame:
        missing = sorted(set(self.selected_features) - set(applicants.columns))
        if missing:
            raise ValueError(f"Prediction input is missing features: {missing}")
        transformed = self.bundle["preprocessor"].transform(applicants)
        raw_probability = self.bundle["model"].predict_proba(transformed)[:, 1]
        calibrator = self.bundle.get("calibrator")
        probability = (
            calibrator.transform(raw_probability)
            if calibrator is not None
            else raw_probability
        )
        bands = assign_risk_band(probability, self.bundle["risk_thresholds"])
        return pd.DataFrame(
            {
                "default_probability": probability,
                "risk_score": np.round(probability * 1000).astype(int),
                "risk_band": bands.astype(str),
            },
            index=applicants.index,
        )

    def local_explanation(self, applicants: pd.DataFrame):
        transformed = self.bundle["preprocessor"].transform(applicants)
        return self.bundle["model"].explain_local(
            transformed,
            name="Applicant-level EBM explanation",
        )
