import numpy as np
import pandas as pd

from src.data.preprocessor import engineer_application_features, make_stratified_split


def sample_frame(rows: int = 100) -> pd.DataFrame:
    target = np.tile([0, 0, 0, 1], rows // 4)
    return pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(rows),
            "TARGET": target,
            "DAYS_BIRTH": -np.linspace(20 * 365.25, 70 * 365.25, rows),
            "DAYS_EMPLOYED": np.where(np.arange(rows) % 10 == 0, 365243, -1000),
            "AMT_CREDIT": 200_000.0,
            "AMT_INCOME_TOTAL": 100_000.0,
            "AMT_ANNUITY": 20_000.0,
            "AMT_GOODS_PRICE": 180_000.0,
            "CODE_GENDER": np.where(np.arange(rows) % 2 == 0, "F", "M"),
        }
    )


def test_engineering_replaces_employment_sentinel() -> None:
    result = engineer_application_features(sample_frame())
    assert result["DAYS_EMPLOYED_ANOM"].sum() == 10
    assert result.loc[result["DAYS_EMPLOYED_ANOM"].eq(1), "DAYS_EMPLOYED_CLEAN"].isna().all()
    assert np.isclose(result["CREDIT_INCOME_RATIO"].dropna().iloc[0], 2.0)


def test_split_is_disjoint_and_stratified() -> None:
    result = make_stratified_split(engineer_application_features(sample_frame()))
    assert len(result.X_train) == 70
    assert len(result.X_validation) == 15
    assert len(result.X_test) == 15
    assert set(result.X_train.index).isdisjoint(result.X_validation.index)
    assert set(result.X_train.index).isdisjoint(result.X_test.index)
    assert abs(result.y_train.mean() - 0.25) < 0.02

