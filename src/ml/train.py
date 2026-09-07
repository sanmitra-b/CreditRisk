from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from interpret import __version__ as interpret_version
from interpret.glassbox import ExplainableBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from src.data.preprocessor import (
    EBMFrameTransformer,
    build_linear_preprocessor,
    build_tree_preprocessor,
    engineer_application_features,
    make_stratified_split,
)
from src.ml.evaluate import (
    binary_metrics,
    PlattCalibrator,
    percentile_thresholds,
    risk_band_summary,
    validate_monotonic_bands,
)
from src.utils.config import get_settings
from src.utils.logger import configure_logging, get_logger


LOGGER = get_logger(__name__)


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)}")


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default),
        encoding="utf-8",
    )


def train_logistic(split, model_dir: Path):
    LOGGER.info("Training balanced Logistic Regression")
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_linear_preprocessor(split.X_train)),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    solver="saga",
                    max_iter=300,
                    tol=1e-2,
                    C=0.5,
                    random_state=42,
                ),
            ),
        ]
    )
    started = time.perf_counter()
    pipeline.fit(split.X_train, split.y_train)
    elapsed = time.perf_counter() - started
    probabilities = pipeline.predict_proba(split.X_test)[:, 1]
    joblib.dump(pipeline, model_dir / "logistic_pipeline.joblib", compress=3)
    return binary_metrics(split.y_test, probabilities, elapsed)


def train_lightgbm(split, model_dir: Path, feature_count: int):
    LOGGER.info("Training shadow LightGBM")
    preprocessor = build_tree_preprocessor(split.X_train)
    X_train = preprocessor.fit_transform(split.X_train)
    X_validation = preprocessor.transform(split.X_validation)
    X_test = preprocessor.transform(split.X_test)
    names = preprocessor.get_feature_names_out().tolist()

    model = LGBMClassifier(
        objective="binary",
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    started = time.perf_counter()
    model.fit(
        X_train,
        split.y_train,
        eval_X=X_validation,
        eval_y=split.y_validation,
        eval_metric="auc",
        callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)],
    )
    elapsed = time.perf_counter() - started
    probabilities = model.predict_proba(X_test)[:, 1]
    importance = (
        pd.DataFrame({"feature": names, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    selected = importance.head(feature_count)["feature"].tolist()
    joblib.dump(
        {
            "preprocessor": preprocessor,
            "model": model,
            "feature_names": names,
            "selected_features": selected,
        },
        model_dir / "lightgbm_shadow.joblib",
        compress=3,
    )
    return (
        binary_metrics(split.y_test, probabilities, elapsed),
        selected,
        importance,
    )


def _stratified_sample(
    X: pd.DataFrame,
    y: pd.Series,
    maximum_rows: int | None,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
    if not maximum_rows or len(X) <= maximum_rows:
        return X, y
    fraction = maximum_rows / len(X)
    sampled_indices = (
        pd.DataFrame({"target": y})
        .groupby("target", group_keys=False)
        .sample(frac=fraction, random_state=random_seed)
        .index
    )
    return X.loc[sampled_indices], y.loc[sampled_indices]


def train_ebm(
    split,
    model_dir: Path,
    selected_features: list[str],
    maximum_rows: int | None,
    random_seed: int,
):
    LOGGER.info(
        "Training production EBM on %s selected features%s",
        len(selected_features),
        f" with at most {maximum_rows:,} rows" if maximum_rows else "",
    )
    preprocessor = EBMFrameTransformer(selected_features).fit(split.X_train)
    X_train = preprocessor.transform(split.X_train)
    X_validation = preprocessor.transform(split.X_validation)
    X_test = preprocessor.transform(split.X_test)
    X_fit, y_fit = _stratified_sample(
        X_train,
        split.y_train,
        maximum_rows=maximum_rows,
        random_seed=random_seed,
    )

    model = ExplainableBoostingClassifier(
        feature_names=selected_features,
        feature_types=preprocessor.ebm_feature_types,
        outer_bags=4,
        inner_bags=0,
        max_bins=128,
        max_interaction_bins=32,
        interactions=8,
        validation_size=0.15,
        max_rounds=15_000,
        early_stopping_rounds=75,
        random_state=random_seed,
        n_jobs=-2,
    )
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_fit)
    started = time.perf_counter()
    model.fit(X_fit, y_fit, sample_weight=sample_weight)
    elapsed = time.perf_counter() - started

    raw_validation_probabilities = model.predict_proba(X_validation)[:, 1]
    raw_test_probabilities = model.predict_proba(X_test)[:, 1]
    calibrator = PlattCalibrator()
    calibrator.fit(raw_validation_probabilities, split.y_validation)
    validation_probabilities = calibrator.transform(raw_validation_probabilities)
    test_probabilities = calibrator.transform(raw_test_probabilities)
    thresholds = percentile_thresholds(validation_probabilities)
    band_summary = risk_band_summary(split.y_test, test_probabilities, thresholds)
    metrics = binary_metrics(
        split.y_test,
        test_probabilities,
        elapsed,
        ranking_probabilities=raw_test_probabilities,
    )
    metrics["training_rows"] = int(len(X_fit))
    metrics["risk_bands_monotonic"] = validate_monotonic_bands(band_summary)

    bundle = {
        "model": model,
        "calibrator": calibrator,
        "preprocessor": preprocessor,
        "selected_features": selected_features,
        "feature_types": preprocessor.ebm_feature_types,
        "risk_thresholds": thresholds,
        "metadata": {
            "training_rows": len(X_fit),
            "random_seed": random_seed,
            "target_definition": "Home Credit TARGET=1 payment difficulty",
            "risk_band_disclaimer": "Internal model-estimated risk segment; not a regulatory credit rating.",
        },
    }
    joblib.dump(bundle, model_dir / "ebm_bundle.joblib", compress=3)
    _save_json(model_dir / "risk_thresholds.json", thresholds)
    return metrics, band_summary, model


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Train the three-model credit-risk stack.")
    parser.add_argument(
        "--features",
        type=Path,
        default=settings.artifact_dir / "applicant_features.parquet",
    )
    parser.add_argument("--top-features", type=int, default=30)
    parser.add_argument(
        "--ebm-max-rows",
        type=int,
        default=120_000,
        help="Maximum stratified EBM training rows; use 0 for the full training split.",
    )
    parser.add_argument(
        "--stage",
        choices=["all", "logistic"],
        default="all",
        help="Run the complete stack or refresh only the Logistic Regression baseline.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir = settings.artifact_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading feature artifact from %s", args.features)
    frame = pd.read_parquet(args.features)
    frame = engineer_application_features(frame)
    split = make_stratified_split(frame, random_seed=settings.random_seed)
    LOGGER.info(
        "Split sizes: train=%s, validation=%s, test=%s",
        len(split.X_train),
        len(split.X_validation),
        len(split.X_test),
    )

    metrics_path = evaluation_dir / "metrics.json"
    if args.stage == "logistic":
        results = (
            json.loads(metrics_path.read_text(encoding="utf-8"))
            if metrics_path.exists()
            else {}
        )
        results["logistic_regression"] = train_logistic(
            split,
            settings.model_dir,
        )
        _save_json(metrics_path, results)
        pd.DataFrame(results).T.to_csv(evaluation_dir / "model_comparison.csv")
        LOGGER.info(
            "Logistic Regression refresh complete. Metrics: %s",
            json.dumps(results["logistic_regression"], indent=2),
        )
        return

    results = {}
    results["logistic_regression"] = train_logistic(split, settings.model_dir)
    lightgbm_metrics, selected_features, importance = train_lightgbm(
        split,
        settings.model_dir,
        args.top_features,
    )
    results["lightgbm_shadow"] = lightgbm_metrics
    importance.to_csv(evaluation_dir / "lightgbm_feature_importance.csv", index=False)

    ebm_metrics, bands, _ = train_ebm(
        split,
        settings.model_dir,
        selected_features,
        maximum_rows=args.ebm_max_rows or None,
        random_seed=settings.random_seed,
    )
    results["ebm"] = ebm_metrics
    bands.to_csv(evaluation_dir / "risk_band_summary.csv", index=False)

    manifest = {
        "selected_features": selected_features,
        "all_input_features": split.X_train.columns.tolist(),
        "split": {
            "train_rows": len(split.X_train),
            "validation_rows": len(split.X_validation),
            "test_rows": len(split.X_test),
            "random_seed": settings.random_seed,
        },
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "lightgbm": lgb.__version__,
            "interpret": interpret_version,
        },
    }
    _save_json(settings.model_dir / "feature_manifest.json", manifest)
    _save_json(metrics_path, results)
    pd.DataFrame(results).T.to_csv(evaluation_dir / "model_comparison.csv")
    LOGGER.info("Training complete. Metrics: %s", json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
