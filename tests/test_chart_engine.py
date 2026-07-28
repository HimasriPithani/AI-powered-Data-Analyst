import pandas as pd
import pytest

from app.core.chart_engine import ChartError, build_chart


def make_df():
    return pd.DataFrame({
        "region": ["N", "S", "E", "N"],
        "revenue": [100, 200, 150, 50],
    })


def test_bar_chart_builds():
    fig = build_chart(make_df(), "bar", x="region", y="revenue")
    assert fig is not None
    assert len(fig.data) >= 1


def test_line_chart_builds():
    fig = build_chart(make_df(), "line", x="region", y="revenue")
    assert fig is not None


def test_pie_chart_builds():
    fig = build_chart(make_df(), "pie", x="region", y="revenue")
    assert fig is not None


def test_unsupported_chart_type_raises():
    with pytest.raises(ChartError):
        build_chart(make_df(), "sankey", x="region", y="revenue")


def test_missing_column_raises():
    with pytest.raises(ChartError):
        build_chart(make_df(), "bar", x="nonexistent", y="revenue")


def test_bar_chart_aggregates_by_sum():
    fig = build_chart(make_df(), "bar", x="region", y="revenue", agg="sum")
    # region 'N' appears twice (100+50=150) so aggregation should have merged it
    total_points = sum(len(trace.x) for trace in fig.data)
    assert total_points == 3  # N, S, E after aggregation
