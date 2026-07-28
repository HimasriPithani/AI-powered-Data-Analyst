"""
Defines the tool schemas exposed to the LLM (OpenAI-style function-calling
format, used by Groq's API) and the dispatcher that executes a requested
tool call against the current session's DataManager, returning a
JSON-serializable result.

This is the core of the "agentic" design: the model decides *which* tool to
call and with what arguments; this module is purely mechanical execution +
safety enforcement, never trusting the LLM's code/SQL/args at face value.
"""
from __future__ import annotations

import json
from typing import Any, Dict

from app.core.anomaly import detect_anomalies
from app.core.chart_engine import ChartError, build_chart
from app.core.data_manager import DataManager
from app.core.sandbox import SandboxError, run_pandas_code
from app.core.sql_engine import SQLExecutionError, run_sql
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _fn(name: str, description: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap a schema in the OpenAI/Groq `{"type": "function", "function": {...}}` shape."""
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": parameters},
    }


TOOL_SCHEMAS = [
    _fn(
        "run_pandas_code",
        "Execute a pandas snippet against the loaded dataframe(s) to answer an "
        "analytical question (aggregations, filtering, ranking, growth rates, "
        "pivoting, top-N, etc). Reference dataframes by their dataset name "
        "(shown in the schema context) as local variables. You MUST assign the "
        "final answer to a variable named `result` (a DataFrame, Series, or scalar).",
        {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python/pandas code. Must set `result = ...` at the end.",
                }
            },
            "required": ["code"],
        },
    ),
    _fn(
        "run_sql",
        "Execute a read-only SQL SELECT query against the loaded dataset(s), each "
        "queryable as a table using its dataset name. Use this when the user "
        "explicitly asks for SQL, or when a SQL query is the clearest way to express "
        "the analysis. Only SELECT/WITH statements are permitted.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "A SQL SELECT statement."}
            },
            "required": ["query"],
        },
    ),
    _fn(
        "create_chart",
        "Generate a chart from a dataset to visualize a trend or comparison. "
        "Aggregation (sum/mean/count/etc) is applied automatically for bar/pie charts "
        "when x, y, and agg are given.",
        {
            "type": "object",
            "properties": {
                "dataset": {"type": "string", "description": "Name of the dataset to chart."},
                "chart_type": {
                    "type": "string",
                    "enum": ["bar", "line", "pie", "scatter", "histogram", "box"],
                },
                "x": {"type": "string", "description": "Column for the x-axis / categories."},
                "y": {"type": "string", "description": "Column for the y-axis / values."},
                "color": {"type": "string", "description": "Optional column to group/color by."},
                "agg": {
                    "type": "string",
                    "enum": ["sum", "mean", "count", "max", "min", "median"],
                    "description": "Aggregation applied for bar/pie charts. Default 'sum'.",
                },
                "title": {"type": "string", "description": "Chart title."},
            },
            "required": ["dataset", "chart_type"],
        },
    ),
    _fn(
        "detect_anomalies",
        "Detect statistical anomalies/outliers in a numeric column using combined "
        "z-score and IQR methods, with an explanation for why each row was flagged. "
        "Optionally detect anomalies within groups (e.g. per region) instead of globally.",
        {
            "type": "object",
            "properties": {
                "dataset": {"type": "string"},
                "column": {"type": "string", "description": "Numeric column to check."},
                "group_by": {
                    "type": "string",
                    "description": "Optional categorical column to detect anomalies within each group.",
                },
                "z_thresh": {"type": "number", "description": "Z-score threshold, default 3.0."},
            },
            "required": ["dataset", "column"],
        },
    ),
    _fn(
        "get_dataset_info",
        "Retrieve schema, dtypes, null counts, numeric summary stats, and a sample "
        "of rows for a dataset. Use this if you need more detail than the initial "
        "schema context to decide how to answer.",
        {
            "type": "object",
            "properties": {"dataset": {"type": "string"}},
            "required": ["dataset"],
        },
    ),
]


def execute_tool(tool_name: str, tool_input: Dict[str, Any], dm: DataManager) -> Dict[str, Any]:
    """Dispatch a single tool call to its implementation. Always returns a
    JSON-serializable dict; errors are captured and returned as data rather
    than raised, so the agent loop can feed them back to the LLM to retry."""
    try:
        if tool_name == "run_pandas_code":
            return run_pandas_code(tool_input["code"], dm.all_frames())

        if tool_name == "run_sql":
            return run_sql(tool_input["query"], dm.all_frames())

        if tool_name == "create_chart":
            df = dm.get(tool_input["dataset"])
            if df is None:
                return {"success": False, "error": f"Unknown dataset '{tool_input['dataset']}'"}
            fig = build_chart(
                df,
                chart_type=tool_input.get("chart_type"),
                x=tool_input.get("x"),
                y=tool_input.get("y"),
                color=tool_input.get("color"),
                agg=tool_input.get("agg", "sum"),
                title=tool_input.get("title"),
            )
            return {
                "success": True,
                "chart_ready": True,
                "figure_json": fig.to_json(),
                "spec": tool_input,
            }

        if tool_name == "detect_anomalies":
            df = dm.get(tool_input["dataset"])
            if df is None:
                return {"success": False, "error": f"Unknown dataset '{tool_input['dataset']}'"}
            return detect_anomalies(
                df,
                column=tool_input["column"],
                group_by=tool_input.get("group_by"),
                z_thresh=tool_input.get("z_thresh", 3.0),
            )

        if tool_name == "get_dataset_info":
            profile = dm.profiles.get(tool_input["dataset"])
            if profile is None:
                return {"success": False, "error": f"Unknown dataset '{tool_input['dataset']}'"}
            return {
                "success": True,
                "n_rows": profile.n_rows,
                "n_cols": profile.n_cols,
                "columns": profile.columns,
                "dtypes": profile.dtypes,
                "null_counts": profile.null_counts,
                "numeric_summary": profile.numeric_summary,
                "sample": profile.sample,
                "quality_warnings": profile.quality_warnings,
            }

        return {"success": False, "error": f"Unknown tool '{tool_name}'"}

    except (SandboxError, SQLExecutionError, ChartError) as e:
        logger.warning(f"Tool '{tool_name}' raised a handled error: {e}")
        return {"success": False, "error": str(e)}
    except Exception as e:  # noqa: BLE001 - never let a tool crash the agent loop
        logger.exception(f"Tool '{tool_name}' raised an unexpected error")
        return {"success": False, "error": f"Unexpected error: {e}"}


def safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, default=str)[:8000]
    except Exception:
        return str(obj)[:8000]
