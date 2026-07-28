import pandas as pd

from app.core.sandbox import run_pandas_code


def make_frames():
    return {"sales": pd.DataFrame({"region": ["N", "S", "N"], "revenue": [100, 200, 50]})}


def test_simple_aggregation():
    out = run_pandas_code("result = sales['revenue'].sum()", make_frames())
    assert out["success"] is True
    assert out["result"] == 350


def test_groupby_returns_dataframe():
    code = "result = sales.groupby('region', as_index=False)['revenue'].sum()"
    out = run_pandas_code(code, make_frames())
    assert out["success"] is True
    assert out["result"]["type"] == "dataframe"
    assert set(r["region"] for r in out["result"]["rows"]) == {"N", "S"}


def test_blocks_import_statements():
    out = run_pandas_code("import os\nresult = 1", make_frames())
    assert out["success"] is False
    assert "not allowed" in out["error"].lower() or "import" in out["error"].lower()


def test_blocks_dangerous_builtins():
    out = run_pandas_code("result = open('/etc/passwd').read()", make_frames())
    assert out["success"] is False


def test_syntax_error_is_handled_gracefully():
    out = run_pandas_code("result = sales[[['bad syntax", make_frames())
    assert out["success"] is False
    assert "syntax" in out["error"].lower()


def test_runtime_error_is_captured_not_raised():
    out = run_pandas_code("result = sales['nonexistent_column']", make_frames())
    assert out["success"] is False
    assert out["error"] is not None
