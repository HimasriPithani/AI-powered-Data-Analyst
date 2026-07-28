"""
LLM client with efficient conversation memory and tool-calling.

Optimizations:
- Keeps only recent conversation history.
- Summarizes tool outputs before replaying them.
- Prevents prompt explosion (Groq 413).
- Supports charts, SQL, pandas, anomaly detection.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import groq

from app.config import settings
from app.core.data_manager import DataManager
from app.core.tools import TOOL_SCHEMAS, execute_tool, safe_json
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# Conversation Limits
# ============================================================

MAX_HISTORY_MESSAGES = 6          # last assistant/user/tool messages
MAX_TOOL_ROWS = 20                # never replay more than this
MAX_TOOL_TEXT = 1000              # chars
MAX_SCHEMA_COLUMNS = 50


SYSTEM_PROMPT = """
You are an expert business data analyst.

You are connected to real CSV datasets through tools.

IMPORTANT RULES

1. NEVER invent values.
2. Always use tools when numbers are required.
3. Keep answers concise.
4. After every tool call, explain the result in simple business language.
5. Never expose internal JSON.
6. Never repeat previous tool outputs.
7. Never request the user to upload data if datasets already exist.
8. If a chart was created successfully, simply explain it instead of recreating it.

When writing pandas code:

- Store the final answer in a variable named `result`.
- Keep code minimal.
- Handle NaN values safely.

Use charts whenever visualization helps.

Current datasets:

{schema_context}
"""


# ============================================================
# Response Models
# ============================================================

@dataclass
class AgentStep:
    tool_name: str
    tool_input: Dict[str, Any]
    tool_result: Dict[str, Any]


@dataclass
class AgentResponse:
    final_text: str
    steps: List[AgentStep] = field(default_factory=list)
    raw_messages: List[Dict[str, Any]] = field(default_factory=list)


# ============================================================
# LLM Client
# ============================================================

class LLMClient:

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.api_key = api_key or settings.groq_api_key
        self.model = model or settings.model_name

        if not self.api_key:
            raise ValueError(
                "Missing GROQ_API_KEY. Configure it in .env or provide it explicitly."
            )

        self.client = groq.Groq(api_key=self.api_key)

    # --------------------------------------------------------
    # Trim old history
    # --------------------------------------------------------

    def _trim_history(
        self,
        history: Optional[List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:

        if not history:
            return []

        return history[-MAX_HISTORY_MESSAGES:]

    # --------------------------------------------------------
    # Reduce tool output before replaying to LLM
    # --------------------------------------------------------

    def _summarize_tool_result(
        self,
        tool_name: str,
        result: Dict[str, Any],
    ) -> Dict[str, Any]:

        if not isinstance(result, dict):
            return {"success": False}

        summary = {
            "success": result.get("success", True)
        }

        # chart
        if tool_name == "create_chart":
            summary["chart_created"] = True
            summary["chart_type"] = (
                result.get("spec", {}).get("chart_type")
                if isinstance(result.get("spec"), dict)
                else None
            )
            return summary

        # dataframe
        if "rows" in result:

            rows = result["rows"]

            if isinstance(rows, list):
                summary["rows_returned"] = len(rows)
                summary["preview"] = rows[:5]

            return summary

        # anomalies
        if "n_anomalies" in result:

            summary["n_anomalies"] = result["n_anomalies"]

            if "column" in result:
                summary["column"] = result["column"]

            return summary

        # plain text summary
        if "summary" in result:
            summary["summary"] = str(result["summary"])[:MAX_TOOL_TEXT]

        elif "message" in result:
            summary["summary"] = str(result["message"])[:MAX_TOOL_TEXT]

        return summary