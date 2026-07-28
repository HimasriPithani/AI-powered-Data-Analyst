"""
Statistical anomaly detection with explanations for *why* each row was
flagged. Two complementary detectors are combined:

  - Z-score:   flags points far from the column mean (good for roughly
               normal distributions).
  - IQR:       flags points outside 1.5x the interquartile range (robust
               to skew/outliers in the reference distribution itself).

A row is flagged if EITHER method fires, and the explanation names which
method(s) fired and by how much, so the reasoning is transparent to the
end user (not just a black-box "this is weird").
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.utils.logger import get_logger

logger = get_logger(__name__)


def detect_anomalies(
    df: pd.DataFrame,
    column: str,
    z_thresh: float = 3.0,
    group_by: Optional[str] = None,
    max_results: int = 50,
) -> Dict[str, Any]:
    if column not in df.columns:
        return {"success": False, "error": f"Column '{column}' not found."}
    if not pd.api.types.is_numeric_dtype(df[column]):
        return {"success": False, "error": f"Column '{column}' is not numeric."}

    work = df.copy()
    work["_row_id"] = work.index

    def _flag_group(g: pd.DataFrame) -> pd.DataFrame:
        col = g[column].astype(float)
        mean, std = col.mean(), col.std(ddof=0)
        q1, q3 = col.quantile(0.25), col.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr

        z = (col - mean) / std if std > 0 else pd.Series(0, index=col.index)
        g = g.copy()
        g["_zscore"] = z
        g["_z_flag"] = z.abs() > z_thresh
        g["_iqr_flag"] = (col < lo) | (col > hi)
        g["_lower_bound"] = lo
        g["_upper_bound"] = hi
        g["_mean"] = mean
        g["_std"] = std
        return g

    if group_by and group_by in df.columns:
        flagged = work.groupby(group_by, group_keys=False).apply(
            _flag_group, include_groups=True
        )
    else:
        flagged = _flag_group(work)

    anomalies = flagged[flagged["_z_flag"] | flagged["_iqr_flag"]].copy()
    anomalies = anomalies.sort_values(
        by="_zscore", key=lambda s: s.abs(), ascending=False
    )

    explanations: List[Dict[str, Any]] = []
    for _, row in anomalies.head(max_results).iterrows():
        reasons = []
        if bool(row["_z_flag"]):
            reasons.append(f"z-score {row['_zscore']:.2f} exceeds threshold {z_thresh}")
        if bool(row["_iqr_flag"]):
            reasons.append(
                f"value {row[column]:.2f} falls outside IQR bounds "
                f"[{row['_lower_bound']:.2f}, {row['_upper_bound']:.2f}]"
            )
        explanations.append({
            "row_id": int(row["_row_id"]),
            "value": float(row[column]),
            "group": row[group_by] if group_by and group_by in row else None,
            "reasons": reasons,
            "context": {
                k: (None if pd.isna(v) else v)
                for k, v in row.drop(
                    labels=[c for c in row.index if c.startswith("_")], errors="ignore"
                ).items()
            },
        })

    return {
        "success": True,
        "column": column,
        "group_by": group_by,
        "total_rows_checked": int(len(df)),
        "n_anomalies": int(len(anomalies)),
        "method": "z-score (|z| > {}) OR IQR (1.5x)".format(z_thresh),
        "anomalies": explanations,
    }
