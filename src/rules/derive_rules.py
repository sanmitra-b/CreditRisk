from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor, export_text

from src.ml.evaluate import assign_risk_band


PROTECTED_OR_POLICY_SENSITIVE = {
    "CODE_GENDER",
    "AGE_YEARS",
    "DAYS_BIRTH",
    "NAME_FAMILY_STATUS",
    "NAME_EDUCATION_TYPE",
}


@dataclass
class SurrogateResult:
    model: DecisionTreeRegressor
    features: list[str]
    rule_text: str
    validation_r2: float
    risk_band_agreement: float


def derive_surrogate_rules(
    X_train: pd.DataFrame,
    ebm_train_probability: np.ndarray,
    X_validation: pd.DataFrame,
    ebm_validation_probability: np.ndarray,
    thresholds: dict[str, float],
    candidate_features: list[str],
    random_seed: int = 42,
) -> SurrogateResult:
    features = [
        feature
        for feature in candidate_features
        if feature not in PROTECTED_OR_POLICY_SENSITIVE
        and pd.api.types.is_numeric_dtype(X_train[feature])
    ]
    if not features:
        raise ValueError("No policy-safe numeric features are available for the surrogate")

    train = X_train[features].replace([np.inf, -np.inf], np.nan)
    validation = X_validation[features].replace([np.inf, -np.inf], np.nan)
    medians = train.median()
    train = train.fillna(medians)
    validation = validation.fillna(medians)

    model = DecisionTreeRegressor(
        max_depth=4,
        min_samples_leaf=0.02,
        random_state=random_seed,
    )
    model.fit(train, ebm_train_probability)
    surrogate_probability = np.clip(model.predict(validation), 0, 1)
    r2 = r2_score(ebm_validation_probability, surrogate_probability)
    ebm_bands = assign_risk_band(ebm_validation_probability, thresholds).astype(str)
    surrogate_bands = assign_risk_band(surrogate_probability, thresholds).astype(str)
    agreement = float(np.mean(ebm_bands == surrogate_bands))
    rules = export_text(model, feature_names=features, decimals=4)
    return SurrogateResult(
        model=model,
        features=features,
        rule_text=rules,
        validation_r2=float(r2),
        risk_band_agreement=agreement,
    )

