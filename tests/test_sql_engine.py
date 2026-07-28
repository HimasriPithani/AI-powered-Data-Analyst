import pandas as pd
import pytest

from app.core.sql_engine import SQLExecutionError, run_sql


def make_frames():
    return {"sales": pd.DataFrame({"region": ["N", "S", "N"], "revenue": [100, 200, 50]})}


def test_basic_select():
    out = run_sql("SELECT region, SUM(revenue) as total FROM sales GROUP BY region", make_frames())
    assert out["success"] is True
    assert out["columns"] == ["region", "total"]
    assert len(out["rows"]) == 2


def test_blocks_ddl_statements():
    with pytest.raises(SQLExecutionError):
        run_sql("DROP TABLE sales", make_frames())


def test_blocks_dml_statements():
    with pytest.raises(SQLExecutionError):
        run_sql("DELETE FROM sales WHERE region = 'N'", make_frames())


def test_blocks_non_select_start():
    with pytest.raises(SQLExecutionError):
        run_sql("PRAGMA table_info(sales)", make_frames())


def test_invalid_sql_raises_readable_error():
    with pytest.raises(SQLExecutionError):
        run_sql("SELECT * FROM nonexistent_table", make_frames())


def test_with_clause_allowed():
    query = "WITH t AS (SELECT * FROM sales) SELECT COUNT(*) as n FROM t"
    out = run_sql(query, make_frames())
    assert out["success"] is True
    assert out["rows"][0]["n"] == 3
