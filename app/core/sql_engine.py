"""
Executes LLM-generated SQL against the in-memory dataframes using DuckDB.
DuckDB can query pandas DataFrames directly (zero-copy), which gives us a
real, standards-compliant SQL surface without needing to load a database
server — and it's inherently safer than exec()'ing code since SQL has no
access to the filesystem, network, or Python runtime.
"""
from __future__ import annotations

from typing import Any, Dict

import duckdb
import numpy as np
import pandas as pd

from app.utils.logger import get_logger

logger = get_logger(__name__)

# Statements that must never reach DuckDB in this read-only analytics tool.
_BLOCKED_KEYWORDS = (
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "ATTACH",
    "COPY", "PRAGMA", "INSTALL", "LOAD", "EXPORT", "IMPORT",
)


class SQLExecutionError(Exception):
    pass


def run_sql(query: str, frames: Dict[str, pd.DataFrame], max_rows: int = 200) -> Dict[str, Any]:
    """Runs a read-only SQL query against the given named dataframes (each
    registered as a DuckDB view under its dataset key)."""
    upper = query.upper()
    if any(kw in upper for kw in _BLOCKED_KEYWORDS):
        raise SQLExecutionError(
            "Only read-only SELECT queries are permitted (DDL/DML is blocked)."
        )
    if not upper.strip().startswith("SELECT") and not upper.strip().startswith("WITH"):
        raise SQLExecutionError("Only SELECT (or WITH ... SELECT) statements are permitted.")

    con = duckdb.connect(database=":memory:")
    try:
        for name, df in frames.items():
            con.register(name, df)
        result_df = con.execute(query).fetchdf()
    except Exception as e:  # noqa: BLE001 - surfaced back to the caller/LLM
        raise SQLExecutionError(f"SQL error: {e}") from e
    finally:
        con.close()

    truncated_df = result_df.head(max_rows)
    return {
        "success": True,
        "shape": list(result_df.shape),
        "columns": list(result_df.columns.astype(str)),
        "rows": truncated_df.replace({np.nan: None}).to_dict(orient="records"),
        "truncated": len(result_df) > max_rows,
    }
