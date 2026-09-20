from pathlib import Path

import pandas as pd

from customerflow.monitoring import drift_report


def _frame(multiplier: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame({
        "Avg. Session Length": [30 + i * 0.2 * multiplier for i in range(30)],
        "Time on App": [10 + i * 0.1 * multiplier for i in range(30)],
        "Time on Website": [35 + i * 0.15 * multiplier for i in range(30)],
        "Length of Membership": [1 + i * 0.08 * multiplier for i in range(30)],
    })


def test_identical_population_is_stable(tmp_path: Path) -> None:
    ref = tmp_path / "reference.csv"
    cur = tmp_path / "current.csv"
    _frame().to_csv(ref, index=False)
    _frame().to_csv(cur, index=False)
    report = drift_report(ref, cur)
    assert report["overall_status"] == "stable"
    assert all(item["psi"] == 0 for item in report["features"].values())


def test_shifted_population_is_detected(tmp_path: Path) -> None:
    ref = tmp_path / "reference.csv"
    cur = tmp_path / "current.csv"
    _frame().to_csv(ref, index=False)
    shifted = _frame()
    shifted["Time on App"] += 25
    shifted.to_csv(cur, index=False)
    report = drift_report(ref, cur)
    assert report["features"]["time_on_app"]["status"] == "critical"
