import pandas as pd

from app.core.data_manager import DataManager
from app.core.tools import execute_tool


def make_dm():
    dm = DataManager()
    dm.frames["sales"] = pd.DataFrame({
        "region": ["N", "S", "N", "E"],
        "revenue": [100, 200, 150, 50],
    })
    # populate a minimal profile so get_dataset_info works
    dm.profiles["sales"] = dm._profile("sales", dm.frames["sales"])
    return dm


def test_execute_run_pandas_code():
    dm = make_dm()
    result = execute_tool("run_pandas_code", {"code": "result = sales['revenue'].sum()"}, dm)
    assert result["success"] is True
    assert result["result"] == 500


def test_execute_run_sql():
    dm = make_dm()
    result = execute_tool("run_sql", {"query": "SELECT COUNT(*) as n FROM sales"}, dm)
    assert result["success"] is True
    assert result["rows"][0]["n"] == 4


def test_execute_create_chart():
    dm = make_dm()
    result = execute_tool(
        "create_chart",
        {"dataset": "sales", "chart_type": "bar", "x": "region", "y": "revenue"},
        dm,
    )
    assert result["success"] is True
    assert result["chart_ready"] is True


def test_execute_detect_anomalies():
    dm = make_dm()
    result = execute_tool(
        "detect_anomalies", {"dataset": "sales", "column": "revenue"}, dm
    )
    assert result["success"] is True


def test_execute_unknown_dataset():
    dm = make_dm()
    result = execute_tool(
        "create_chart",
        {"dataset": "nope", "chart_type": "bar", "x": "region", "y": "revenue"},
        dm,
    )
    assert result["success"] is False


def test_execute_unknown_tool():
    dm = make_dm()
    result = execute_tool("not_a_real_tool", {}, dm)
    assert result["success"] is False


def test_get_dataset_info():
    dm = make_dm()
    result = execute_tool("get_dataset_info", {"dataset": "sales"}, dm)
    assert result["success"] is True
    assert "region" in result["columns"]
