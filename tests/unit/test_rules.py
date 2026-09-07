import numpy as np
import pandas as pd

from src.rules.derive_rules import derive_surrogate_rules


def test_surrogate_excludes_sensitive_features() -> None:
    X = pd.DataFrame(
        {
            "AGE_YEARS": np.linspace(20, 70, 100),
            "EXT_SOURCE_3": np.linspace(0, 1, 100),
            "ANNUITY_INCOME_RATIO": np.linspace(0.1, 0.5, 100),
        }
    )
    probability = 0.05 + 0.2 * (1 - X["EXT_SOURCE_3"].to_numpy())
    result = derive_surrogate_rules(
        X,
        probability,
        X,
        probability,
        {"low_medium": 0.10, "medium_high": 0.18},
        X.columns.tolist(),
    )
    assert "AGE_YEARS" not in result.features
    assert "EXT_SOURCE_3" in result.features

