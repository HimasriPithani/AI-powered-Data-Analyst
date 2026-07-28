"""
Exports the current chat session (questions, answers, and chart specs) as a
self-contained Markdown report the user can download and share — a simple
but genuinely useful "export report" bonus feature.
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List


def build_markdown_report(
    dataset_names: List[str],
    chat_history: List[Dict[str, Any]],
) -> str:
    lines = [
        f"# AI Data Analyst — Session Report",
        f"_Generated {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}_",
        "",
        f"**Datasets analyzed:** {', '.join(dataset_names) if dataset_names else 'none'}",
        "",
        "---",
        "",
    ]

    for i, turn in enumerate(chat_history, start=1):
        role = turn.get("role")
        if role == "user":
            lines.append(f"### Q{i}: {turn.get('content')}")
        elif role == "assistant":
            lines.append(turn.get("content", ""))
            charts = turn.get("charts") or []
            for c in charts:
                lines.append(f"\n_Chart generated: {c}_")
        lines.append("")

    return "\n".join(lines)
