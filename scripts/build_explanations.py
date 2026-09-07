from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import shap

from src.data.preprocessor import engineer_application_features, make_stratified_split
from src.ml.evaluate import assign_risk_band
from src.rules.derive_rules import (
    PROTECTED_OR_POLICY_SENSITIVE,
    derive_surrogate_rules,
)
from src.utils.config import get_settings
from src.utils.logger import configure_logging, get_logger


LOGGER = get_logger(__name__)


def _json_value(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    frame = engineer_application_features(
        pd.read_parquet(settings.artifact_dir / "applicant_features.parquet")
    )
    split = make_stratified_split(frame, random_seed=settings.random_seed)
    ebm_bundle = joblib.load(settings.model_dir / "ebm_bundle.joblib")
    lightgbm_bundle = joblib.load(settings.model_dir / "lightgbm_shadow.joblib")

    ebm_model = ebm_bundle["model"]
    ebm_preprocessor = ebm_bundle["preprocessor"]
    calibrator = ebm_bundle["calibrator"]
    thresholds = ebm_bundle["risk_thresholds"]

    X_train_ebm = ebm_preprocessor.transform(split.X_train)
    X_validation_ebm = ebm_preprocessor.transform(split.X_validation)
    raw_train_probability = ebm_model.predict_proba(X_train_ebm)[:, 1]
    raw_validation_probability = ebm_model.predict_proba(X_validation_ebm)[:, 1]
    train_probability = calibrator.transform(raw_train_probability)
    validation_probability = calibrator.transform(raw_validation_probability)

    global_importance = (
        pd.DataFrame(
            {
                "term": ebm_model.term_names_,
                "importance": ebm_model.term_importances(),
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    global_importance.to_csv(
        settings.artifact_dir / "evaluation" / "ebm_global_importance.csv",
        index=False,
    )

    bands = assign_risk_band(validation_probability, thresholds).astype(str)
    local_rows = []
    for band in ["Low", "Medium", "High"]:
        positions = np.flatnonzero(bands == band)
        if not len(positions):
            continue
        position = int(positions[len(positions) // 2])
        one_row = X_validation_ebm.iloc[[position]]
        explanation = ebm_model.explain_local(one_row).data(0)
        contributions = sorted(
            zip(
                explanation["names"],
                explanation["values"],
                explanation["scores"],
            ),
            key=lambda item: abs(float(item[2])),
            reverse=True,
        )[:10]
        local_rows.append(
            {
                "risk_band": band,
                "estimated_probability": float(validation_probability[position]),
                "top_contributions": [
                    {
                        "term": name,
                        "value": _json_value(value),
                        "score": float(score),
                    }
                    for name, value, score in contributions
                ],
            }
        )
    (settings.artifact_dir / "evaluation" / "ebm_local_examples.json").write_text(
        json.dumps(local_rows, indent=2, default=str),
        encoding="utf-8",
    )

    surrogate = derive_surrogate_rules(
        split.X_train,
        train_probability,
        split.X_validation,
        validation_probability,
        thresholds,
        ebm_bundle["selected_features"],
        random_seed=settings.random_seed,
    )
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": surrogate.model,
            "features": surrogate.features,
            "thresholds": thresholds,
        },
        settings.model_dir / "surrogate_tree.joblib",
        compress=3,
    )
    rules_payload = {
        "validation_r2": surrogate.validation_r2,
        "risk_band_agreement": surrogate.risk_band_agreement,
        "features": surrogate.features,
        "excluded_policy_sensitive_features": sorted(
            PROTECTED_OR_POLICY_SENSITIVE.intersection(
                ebm_bundle["selected_features"]
            )
        ),
        "rules": surrogate.rule_text,
        "disclaimer": "Rules approximate EBM scores for explanation only and are not approval policy.",
    }
    (settings.artifact_dir / "rules" / "business_rules.json").write_text(
        json.dumps(rules_payload, indent=2),
        encoding="utf-8",
    )

    lgb_preprocessor = lightgbm_bundle["preprocessor"]
    lgb_model = lightgbm_bundle["model"]
    sample = split.X_validation.sample(
        n=min(2_000, len(split.X_validation)),
        random_state=settings.random_seed,
    )
    transformed_sample = lgb_preprocessor.transform(sample)
    explainer = shap.TreeExplainer(lgb_model)
    shap_values = np.asarray(explainer.shap_values(transformed_sample))
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, -1]
    mean_absolute = np.abs(shap_values).mean(axis=0)
    shap_summary = (
        pd.DataFrame(
            {
                "feature": lightgbm_bundle["feature_names"],
                "mean_abs_shap": mean_absolute,
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    shap_summary.to_csv(
        settings.artifact_dir / "evaluation" / "lightgbm_shap_summary.csv",
        index=False,
    )
    LOGGER.info(
        "Saved explanations; surrogate R2=%.3f, band agreement=%.1f%%",
        surrogate.validation_r2,
        100 * surrogate.risk_band_agreement,
    )


if __name__ == "__main__":
    main()

