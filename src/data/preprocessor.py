from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


TARGET = "TARGET"
ID_COLUMNS = ("SK_ID_CURR", "SK_ID_PREV", "SK_ID_BUREAU")
DEFAULT_RANDOM_SEED = 42


@dataclass(frozen=True)
class DatasetSplit:
    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    clean_denominator = denominator.replace(0, np.nan)
    return numerator.div(clean_denominator).replace([np.inf, -np.inf], np.nan)


def engineer_application_features(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "AMT_CREDIT",
        "AMT_INCOME_TOTAL",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Cannot engineer application features; missing: {missing}")

    output = frame.copy()
    employment_anomaly = output["DAYS_EMPLOYED"].eq(365243)
    age_years = -output["DAYS_BIRTH"] / 365.25
    employment_days = output["DAYS_EMPLOYED"].mask(employment_anomaly)
    return output.assign(
        AGE_YEARS=age_years,
        DAYS_EMPLOYED_ANOM=employment_anomaly.astype("int8"),
        DAYS_EMPLOYED_CLEAN=employment_days,
        CREDIT_INCOME_RATIO=safe_ratio(
            output["AMT_CREDIT"], output["AMT_INCOME_TOTAL"]
        ),
        ANNUITY_INCOME_RATIO=safe_ratio(
            output["AMT_ANNUITY"], output["AMT_INCOME_TOTAL"]
        ),
        CREDIT_TERM=safe_ratio(output["AMT_ANNUITY"], output["AMT_CREDIT"]),
        CREDIT_GOODS_RATIO=safe_ratio(
            output["AMT_CREDIT"], output["AMT_GOODS_PRICE"]
        ),
        EMPLOYED_AGE_RATIO=safe_ratio(-employment_days, age_years * 365.25),
    )


def make_stratified_split(
    frame: pd.DataFrame,
    target: str = TARGET,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> DatasetSplit:
    if target not in frame:
        raise ValueError(f"Target column {target!r} is missing")
    if frame[target].isna().any():
        raise ValueError("Target contains missing values")

    excluded = {target, *ID_COLUMNS}
    feature_columns = [column for column in frame.columns if column not in excluded]
    X = frame[feature_columns].copy()
    y = frame[target].astype("int8")

    X_train, X_remaining, y_train, y_remaining = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=random_seed,
    )
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_remaining,
        y_remaining,
        test_size=0.50,
        stratify=y_remaining,
        random_state=random_seed,
    )
    return DatasetSplit(
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,
        y_train=y_train,
        y_validation=y_validation,
        y_test=y_test,
    )


def feature_types(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = frame.select_dtypes(include="number").columns.tolist()
    categorical = [column for column in frame.columns if column not in numeric]
    return numeric, categorical


def build_linear_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    numeric, categorical = feature_types(frame)
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="__MISSING__"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=100,
                    sparse_output=True,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        sparse_threshold=0.3,
    )


def build_tree_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    numeric, categorical = feature_types(frame)
    return ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median", add_indicator=False), numeric),
            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="constant",
                                fill_value="__MISSING__",
                            ),
                        ),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                                encoded_missing_value=-1,
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        verbose_feature_names_out=False,
    )


class EBMFrameTransformer(BaseEstimator, TransformerMixin):
    """Persistable train-fitted cleaning for mixed-type EBM inputs."""

    def __init__(self, features: Sequence[str]):
        self.features = list(features)

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        missing = sorted(set(self.features) - set(X.columns))
        if missing:
            raise ValueError(f"EBM features are missing during fit: {missing}")
        selected = X[self.features]
        self.numeric_features_, self.categorical_features_ = feature_types(selected)
        self.numeric_medians_ = (
            selected[self.numeric_features_]
            .replace([np.inf, -np.inf], np.nan)
            .median()
            .to_dict()
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = sorted(set(self.features) - set(X.columns))
        if missing:
            raise ValueError(f"EBM features are missing during transform: {missing}")
        output = X[self.features].copy()
        for column in self.numeric_features_:
            output[column] = (
                pd.to_numeric(output[column], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(self.numeric_medians_[column])
            )
        for column in self.categorical_features_:
            output[column] = output[column].astype("string").fillna("__MISSING__")
        return output

    @property
    def ebm_feature_types(self) -> list[str]:
        categorical = set(self.categorical_features_)
        return ["nominal" if feature in categorical else "continuous" for feature in self.features]

