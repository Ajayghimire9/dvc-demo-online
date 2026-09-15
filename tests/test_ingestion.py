import pandas as pd
import pytest

from customerflow.data import FEATURES, TARGET, prepare_frame
from customerflow.model import train_model
from customerflow.registry import load_bundle


def test_curation_normalises_headers_and_removes_pii():
    frame = pd.DataFrame(
        {
            "Email": ["person@example.test", "other@example.test"],
            "Time on App": [12.0, 13.0],
            "Time on Website": [35.0, 36.0],
            "Avg. Session Length": [30.0, 31.0],
            "Length of Membership": [0.5, 2.0],
            "Yearly Amount Spent": [400.0, 500.0],
        }
    )
    result = prepare_frame(frame, minimum_membership=1.0)
    assert list(result.columns) == FEATURES + [TARGET]
    assert len(result) == 1


def test_curation_rejects_missing_or_invalid_values():
    frame = pd.read_csv("examples/customers.csv")
    with pytest.raises(ValueError, match="missing columns"):
        prepare_frame(frame.drop(columns="Time on App"))
    invalid = frame.copy()
    invalid["Time on App"] = invalid["Time on App"].astype(object)
    invalid.loc[0, "Time on App"] = "unknown"
    with pytest.raises((ValueError, TypeError)):
        prepare_frame(invalid)


def test_train_writes_checksum_verified_bundle(tmp_path):
    metrics = train_model("examples/customers.csv", tmp_path)
    assert metrics["selected_model"] in {
        "baseline_mean",
        "ridge",
        "random_forest",
        "hist_gradient_boosting",
    }
    bundle = load_bundle(tmp_path / "model.joblib", tmp_path / "model.manifest.json")
    assert bundle["features"] == FEATURES
    assert (tmp_path / "metrics.json").is_file()
