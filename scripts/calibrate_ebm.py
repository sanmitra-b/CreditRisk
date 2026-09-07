from __future__ import annotations

import json

import joblib
import pandas as pd
from src.data.preprocessor import engineer_application_features, make_stratified_split
from src.ml.evaluate import (
    PlattCalibrator,
    binary_metrics,
    percentile_thresholds,
    risk_band_summary,
    validate_monotonic_bands,
)
from src.utils.config import get_settings
from src.utils.logger import configure_logging, get_logger


LOGGER = get_logger(__name__)


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    features_path = settings.artifact_dir / "applicant_features.parquet"
    bundle_path = settings.model_dir / "ebm_bundle.joblib"
    frame = engineer_application_features(pd.read_parquet(features_path))
    split = make_stratified_split(frame, random_seed=settings.random_seed)
    bundle = joblib.load(bundle_path)

    preprocessor = bundle["preprocessor"]
    model = bundle["model"]
    X_validation = preprocessor.transform(split.X_validation)
    X_test = preprocessor.transform(split.X_test)
    raw_validation = model.predict_proba(X_validation)[:, 1]
    raw_test = model.predict_proba(X_test)[:, 1]

    calibrator = PlattCalibrator()
    calibrator.fit(raw_validation, split.y_validation)
    validation_probability = calibrator.transform(raw_validation)
    test_probability = calibrator.transform(raw_test)
    thresholds = percentile_thresholds(validation_probability)
    bands = risk_band_summary(split.y_test, test_probability, thresholds)
    metrics = binary_metrics(
        split.y_test,
        test_probability,
        ranking_probabilities=raw_test,
    )
    metrics["training_rows"] = bundle["metadata"]["training_rows"]
    metrics["risk_bands_monotonic"] = validate_monotonic_bands(bands)

    bundle["calibrator"] = calibrator
    bundle["risk_thresholds"] = thresholds
    bundle["metadata"]["calibration"] = "Platt scaling fitted on validation split"
    joblib.dump(bundle, bundle_path, compress=3)

    (settings.model_dir / "risk_thresholds.json").write_text(
        json.dumps(thresholds, indent=2),
        encoding="utf-8",
    )
    bands.to_csv(settings.artifact_dir / "evaluation" / "risk_band_summary.csv", index=False)

    metrics_path = settings.artifact_dir / "evaluation" / "metrics.json"
    all_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    all_metrics["ebm"] = metrics
    metrics_path.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    pd.DataFrame(all_metrics).T.to_csv(
        settings.artifact_dir / "evaluation" / "model_comparison.csv"
    )
    LOGGER.info("Calibrated EBM metrics: %s", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
