"""
Builds Plotly figures from a dataframe based on a declarative spec produced
by the LLM (chart_type, x, y, color, agg, title). Keeping chart construction
declarative (rather than letting the LLM write raw plotting code) keeps this
path fully sandboxed with no code execution.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app.utils.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_TYPES = {"bar", "line", "pie", "scatter", "histogram", "box"}


class ChartError(Exception):
    pass


def build_chart(
    df: pd.DataFrame,
    chart_type: str,
    x: Optional[str] = None,
    y: Optional[str] = None,
    color: Optional[str] = None,
    agg: Optional[str] = "sum",
    title: Optional[str] = None,
) -> go.Figure:
    chart_type = (chart_type or "").lower().strip()
    if chart_type not in SUPPORTED_TYPES:
        raise ChartError(f"Unsupported chart type '{chart_type}'. Supported: {sorted(SUPPORTED_TYPES)}")

    for col in filter(None, [x, y, color]):
        if col not in df.columns:
            raise ChartError(f"Column '{col}' not found in the data.")

    plot_df = df.copy()
    title = title or f"{chart_type.title()} chart"

    if chart_type in {"bar", "pie"} and x and y and agg:
        group_cols = [x] + ([color] if color and color != x else [])
        if agg not in {"sum", "mean", "count", "max", "min", "median"}:
            agg = "sum"
        plot_df = plot_df.groupby(group_cols, as_index=False)[y].agg(agg)
    elif chart_type == "line" and x and y and agg and pd.api.types.is_datetime64_any_dtype(plot_df[x]):
        # Bucket a time series into monthly periods before plotting. Without
        # this, a "trend over time" line chart on daily/transaction-level
        # data is just one noisy point per row instead of a readable trend.
        if agg not in {"sum", "mean", "count", "max", "min", "median"}:
            agg = "sum"
        group_cols = [x] + ([color] if color and color != x else [])
        plot_df[x] = plot_df[x].dt.to_period("M").dt.to_timestamp()
        plot_df = plot_df.groupby(group_cols, as_index=False)[y].agg(agg)

    try:
        if chart_type == "bar":
            fig = px.bar(plot_df, x=x, y=y, color=color, title=title)
        elif chart_type == "line":
            plot_df = plot_df.sort_values(x) if x else plot_df
            fig = px.line(plot_df, x=x, y=y, color=color, title=title, markers=True)
        elif chart_type == "pie":
            fig = px.pie(plot_df, names=x, values=y, title=title)
        elif chart_type == "scatter":
            fig = px.scatter(plot_df, x=x, y=y, color=color, title=title)
        elif chart_type == "histogram":
            fig = px.histogram(plot_df, x=x, color=color, title=title)
        elif chart_type == "box":
            fig = px.box(plot_df, x=x, y=y, color=color, title=title)
        else:  # pragma: no cover - guarded above
            raise ChartError(f"Unsupported chart type '{chart_type}'.")
    except Exception as e:  # noqa: BLE001
        raise ChartError(f"Failed to build chart: {e}") from e

    fig.update_layout(template="plotly_white", margin=dict(l=40, r=20, t=60, b=40))
    return fig


def chart_spec_summary(spec: Dict[str, Any]) -> str:
    """Human/LLM-readable one-liner describing a chart, for chat history."""
    return (
        f"{spec.get('chart_type', '?').title()} chart of "
        f"{spec.get('y', '?')} by {spec.get('x', '?')}"
        + (f", grouped by {spec['color']}" if spec.get("color") else "")
    )