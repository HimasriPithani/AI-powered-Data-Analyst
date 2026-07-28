"""
Canned prompt templates for the one-click UI buttons (Summary, Trends,
Underperformers, Top customers, Anomaly scan). These are just regular
questions routed through the same agent loop as free-text chat — kept here
so the prompt wording is defined once and easy to tune.
"""

QUICK_PROMPTS = {
    "Executive summary": (
        "Give me a concise executive summary of this dataset: key totals, the "
        "overall trend over time if there's a date column, and the 2-3 most "
        "important business insights. Use tools to compute real numbers."
    ),
    "Revenue/trend over time": (
        "Show how the main numeric metric (e.g. revenue) trends over time. "
        "Create a line chart aggregated by month if there's a date column, and "
        "summarize whether it's growing, shrinking, or flat."
    ),
    "Top performers": (
        "Identify the top 5 performers (e.g. top products, regions, or customers "
        "by revenue) and explain what's driving their performance."
    ),
    "Underperformers": (
        "Identify which categories (products/regions/segments) are underperforming "
        "relative to the rest, and quantify the gap."
    ),
    "Detect anomalies": (
        "Scan the main numeric columns for anomalies/outliers, explain why each "
        "flagged value looks unusual, and suggest what might be worth investigating."
    ),
    "Data quality check": (
        "Run a data quality check: missing values, duplicates, suspicious "
        "zero/negative values, and any other issues. Summarize what should be "
        "cleaned before this data is trusted for reporting."
    ),
}
