import io

import pandas as pd
import pytest

from app.core.data_manager import DataManager, DataValidationError


def make_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()


def test_load_valid_csv():
    dm = DataManager()
    df = pd.DataFrame({"Region": ["East", "West"], "Revenue": [100, 200]})
    profile = dm.load_csv("sales.csv", make_csv_bytes(df))
    assert profile.n_rows == 2
    assert "region" in profile.columns
    assert "revenue" in profile.columns


def test_empty_file_raises():
    dm = DataManager()
    with pytest.raises(DataValidationError):
        dm.load_csv("empty.csv", b"")


def test_unparsable_file_raises():
    dm = DataManager()
    with pytest.raises(DataValidationError):
        dm.load_csv("bad.csv", b"\x00\x01\x02\x03not,a,csv\x00")


def test_numeric_type_inference():
    dm = DataManager()
    df = pd.DataFrame({"amount": ["1,000", "2,500", "3,000"]})
    profile = dm.load_csv("amounts.csv", make_csv_bytes(df))
    assert "amount" in profile.numeric_summary


def test_datetime_type_inference():
    dm = DataManager()
    df = pd.DataFrame({"date": ["2024-01-01", "2024-01-02", "2024-01-03"]})
    dm.load_csv("dates.csv", make_csv_bytes(df))
    loaded = dm.get("dates")
    assert pd.api.types.is_datetime64_any_dtype(loaded["date"])


def test_duplicate_rows_flagged():
    dm = DataManager()
    df = pd.DataFrame({"a": [1, 1], "b": [2, 2]})
    profile = dm.load_csv("dup.csv", make_csv_bytes(df))
    assert any("duplicat" in w.lower() for w in profile.quality_warnings)


def test_schema_context_contains_dataset_names():
    dm = DataManager()
    df = pd.DataFrame({"x": [1, 2, 3]})
    dm.load_csv("mydata.csv", make_csv_bytes(df))
    ctx = dm.schema_context()
    assert "mydata" in ctx
