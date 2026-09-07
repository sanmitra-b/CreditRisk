from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.utils.logger import get_logger


LOGGER = get_logger(__name__)
TARGET = "TARGET"
ID_COLUMN = "SK_ID_CURR"


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], table_name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")


def load_application(data_dir: Path, filename: str = "application_train.csv") -> pd.DataFrame:
    path = data_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Application table not found: {path}")
    frame = pd.read_csv(path).copy()
    _require_columns(frame, [ID_COLUMN], filename)
    if frame[ID_COLUMN].duplicated().any():
        raise ValueError(f"{filename} does not contain one row per {ID_COLUMN}")
    return frame


def aggregate_bureau(data_dir: Path) -> pd.DataFrame | None:
    path = data_dir / "bureau.csv"
    if not path.exists():
        LOGGER.warning("Skipping bureau aggregates: %s does not exist", path)
        return None

    usecols = [
        ID_COLUMN,
        "SK_ID_BUREAU",
        "CREDIT_ACTIVE",
        "CREDIT_DAY_OVERDUE",
        "AMT_CREDIT_SUM_OVERDUE",
    ]
    bureau = pd.read_csv(path, usecols=usecols)
    bureau = bureau.assign(
        IS_ACTIVE=bureau["CREDIT_ACTIVE"].eq("Active").astype("int8"),
        HAS_OVERDUE=(
            bureau["CREDIT_DAY_OVERDUE"].gt(0)
            | bureau["AMT_CREDIT_SUM_OVERDUE"].fillna(0).gt(0)
        ).astype("int8"),
    )
    return (
        bureau.groupby(ID_COLUMN, as_index=False)
        .agg(
            PRIOR_CREDIT_COUNT=("SK_ID_BUREAU", "nunique"),
            ACTIVE_CREDIT_COUNT=("IS_ACTIVE", "sum"),
            OVERDUE_SHARE=("HAS_OVERDUE", "mean"),
            MAX_DAYS_OVERDUE=("CREDIT_DAY_OVERDUE", "max"),
        )
        .reset_index(drop=True)
    )


def aggregate_previous_applications(data_dir: Path) -> pd.DataFrame | None:
    path = data_dir / "previous_application.csv"
    if not path.exists():
        LOGGER.warning("Skipping previous-application aggregates: %s does not exist", path)
        return None

    previous = pd.read_csv(
        path,
        usecols=[ID_COLUMN, "SK_ID_PREV", "NAME_CONTRACT_STATUS"],
    )
    previous = previous.assign(
        IS_REFUSED=previous["NAME_CONTRACT_STATUS"].eq("Refused").astype("int8")
    )
    return (
        previous.groupby(ID_COLUMN, as_index=False)
        .agg(
            PREV_APP_COUNT=("SK_ID_PREV", "nunique"),
            PREV_REFUSED_COUNT=("IS_REFUSED", "sum"),
            PREV_REFUSAL_RATE=("IS_REFUSED", "mean"),
        )
        .reset_index(drop=True)
    )


def aggregate_installments(data_dir: Path, chunksize: int = 1_000_000) -> pd.DataFrame | None:
    path = data_dir / "installments_payments.csv"
    if not path.exists():
        LOGGER.warning("Skipping installment aggregates: %s does not exist", path)
        return None

    usecols = [
        ID_COLUMN,
        "DAYS_INSTALMENT",
        "DAYS_ENTRY_PAYMENT",
        "AMT_INSTALMENT",
        "AMT_PAYMENT",
    ]
    dtypes = {
        ID_COLUMN: "int32",
        "DAYS_INSTALMENT": "float32",
        "DAYS_ENTRY_PAYMENT": "float32",
        "AMT_INSTALMENT": "float32",
        "AMT_PAYMENT": "float32",
    }
    partials: list[pd.DataFrame] = []
    for chunk_number, chunk in enumerate(
        pd.read_csv(path, usecols=usecols, dtype=dtypes, chunksize=chunksize),
        start=1,
    ):
        observed = chunk["DAYS_ENTRY_PAYMENT"].notna()
        chunk = chunk.assign(
            OBSERVED_PAYMENT=observed.astype("int8"),
            LATE_PAYMENT=(
                observed
                & chunk["DAYS_ENTRY_PAYMENT"].gt(chunk["DAYS_INSTALMENT"])
            ).astype("int8"),
            UNDERPAYMENT=(
                chunk["AMT_PAYMENT"].notna()
                & chunk["AMT_PAYMENT"].lt(chunk["AMT_INSTALMENT"] * 0.99)
            ).astype("int8"),
        )
        partials.append(
            chunk.groupby(ID_COLUMN, as_index=False).agg(
                INSTALLMENT_ROWS=(ID_COLUMN, "size"),
                OBSERVED_PAYMENT_COUNT=("OBSERVED_PAYMENT", "sum"),
                LATE_PAYMENT_COUNT=("LATE_PAYMENT", "sum"),
                UNDERPAYMENT_COUNT=("UNDERPAYMENT", "sum"),
                SCHEDULED_AMOUNT=("AMT_INSTALMENT", "sum"),
                PAID_AMOUNT=("AMT_PAYMENT", "sum"),
            )
        )
        LOGGER.info("Aggregated installment chunk %s", chunk_number)

    aggregate = pd.concat(partials, ignore_index=True).groupby(
        ID_COLUMN, as_index=False
    ).sum(numeric_only=True)
    observed_count = aggregate["OBSERVED_PAYMENT_COUNT"].replace(0, np.nan)
    scheduled_amount = aggregate["SCHEDULED_AMOUNT"].replace(0, np.nan)
    aggregate["INST_LATE_RATE"] = aggregate["LATE_PAYMENT_COUNT"] / observed_count
    aggregate["INST_UNDERPAY_RATE"] = (
        aggregate["UNDERPAYMENT_COUNT"] / observed_count
    )
    aggregate["INST_PAYMENT_RATIO"] = aggregate["PAID_AMOUNT"] / scheduled_amount
    return aggregate


def _merge_history(
    application: pd.DataFrame,
    aggregate: pd.DataFrame | None,
    source_name: str,
    indicator_name: str,
) -> pd.DataFrame:
    if aggregate is None:
        return application.assign(**{indicator_name: np.int8(0)})
    before = len(application)
    merged = application.merge(
        aggregate,
        on=ID_COLUMN,
        how="left",
        validate="one_to_one",
    )
    if len(merged) != before or merged[ID_COLUMN].duplicated().any():
        raise AssertionError(f"{source_name} merge changed the application grain")
    aggregate_columns = [column for column in aggregate.columns if column != ID_COLUMN]
    merged[indicator_name] = merged[aggregate_columns].notna().any(axis=1).astype("int8")
    LOGGER.info(
        "%s coverage: %.1f%%",
        source_name,
        100 * merged[indicator_name].mean(),
    )
    return merged


def build_applicant_table(
    data_dir: Path,
    include_installments: bool = True,
    installment_chunksize: int = 1_000_000,
) -> pd.DataFrame:
    application = load_application(data_dir)
    application = _merge_history(
        application,
        aggregate_bureau(data_dir),
        "bureau",
        "HAS_BUREAU_HISTORY",
    )
    application = _merge_history(
        application,
        aggregate_previous_applications(data_dir),
        "previous applications",
        "HAS_PREVIOUS_APPLICATION_HISTORY",
    )
    if include_installments:
        application = _merge_history(
            application,
            aggregate_installments(data_dir, installment_chunksize),
            "installments",
            "HAS_INSTALLMENT_HISTORY",
        )
    else:
        application["HAS_INSTALLMENT_HISTORY"] = np.int8(0)
    return application

