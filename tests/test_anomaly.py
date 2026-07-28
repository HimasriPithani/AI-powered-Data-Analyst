import numpy as np
import pandas as pd

from app.core.anomaly import detect_anomalies


def make_df_with_outlier():
    values = [10, 12, 11, 9, 10, 13, 11, 10, 1000]  # last one is a huge outlier
    return pd.DataFrame({"amount": values, "region": ["N"] * 9})


def test_detects_obvious_outlier():
    df = make_df_with_outlier()
    result = detect_anomalies(df, "amount")
    assert result["success"] is True
    assert result["n_anomalies"] >= 1
    flagged_values = [a["value"] for a in result["anomalies"]]
    assert 1000.0 in flagged_values


def test_missing_column_returns_error():
    df = make_df_with_outlier()
    result = detect_anomalies(df, "does_not_exist")
    assert result["success"] is False


def test_non_numeric_column_returns_error():
    df = pd.DataFrame({"name": ["a", "b", "c"]})
    result = detect_anomalies(df, "name")
    assert result["success"] is False


def test_no_anomalies_in_uniform_data():
    df = pd.DataFrame({"amount": [10, 10, 10, 10, 10, 10]})
    result = detect_anomalies(df, "amount")
    assert result["success"] is True
    assert result["n_anomalies"] == 0


def test_group_by_scopes_detection():
    df = pd.DataFrame({
        "amount": [10, 10, 10, 500, 1000, 1010, 990, 1005],
        "region": ["N", "N", "N", "N", "S", "S", "S", "S"],
    })
    result = detect_anomalies(df, "amount", group_by="region")
    assert result["success"] is True
    # 500 should be flagged as an outlier within the "N" group only
    assert result["n_anomalies"] >= 1
