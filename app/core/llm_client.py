"""
Wraps the Groq API (OpenAI-compatible chat completions) and implements the
agent loop:

    user question -> model (with tools) -> [tool_calls?] -> execute tools ->
    feed tool results back as role="tool" messages -> ... -> final text answer

The model decides which tool(s) to call (pandas, SQL, chart, anomaly
detection, dataset info) based on the question; this module just runs the
loop and enforces a max-turn budget so a confused model can't loop forever.
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

SYSTEM_PROMPT = """You are an expert data analyst assistant embedded in a CSV analytics app.

You have access to tools that let you actually query the user's real data — never invent
numbers. Ground every answer in a tool result.

Guidelines:
- Use `run_pandas_code` or `run_sql` for aggregations, filtering, ranking, trends, and
  calculations. Prefer SQL when the user asks for SQL explicitly; otherwise use whichever is
  clearer. For run_pandas_code you MUST assign the answer to a variable named `result`.
- Use `create_chart` whenever a visualization would help answer the question, or when asked
  directly for a chart/graph/plot.
- Use `detect_anomalies` when asked about anomalies, outliers, unusual values, or data quality
  issues, and clearly explain WHY each flagged row is unusual using the reasons returned.
- Use `get_dataset_info` if you need more schema/summary detail before writing code.
- After getting tool results, write a clear, business-friendly final answer: lead with the
  direct answer, then briefly explain your reasoning/method, and mention concrete numbers.
- If a tool call fails, read the error and try a corrected call rather than giving up.
- Keep pandas/SQL code simple, defensive (handle NaNs), and scoped only to what's needed.
- Never fabricate column names — check the schema context / get_dataset_info first if unsure.

Here is the schema of the currently loaded dataset(s):
{schema_context}
"""


@dataclass
class AgentStep:
    """One executed tool call, kept for the UI to render (chart, table, etc)."""
    tool_name: str
    tool_input: Dict[str, Any]
    tool_result: Dict[str, Any]


@dataclass
class AgentResponse:
    final_text: str
    steps: List[AgentStep] = field(default_factory=list)
    raw_messages: List[Dict[str, Any]] = field(default_factory=list)


class LLMClient:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or settings.groq_api_key
        self.model = model or settings.model_name
        if not self.api_key:
            raise ValueError(
                "No Groq API key configured. Set GROQ_API_KEY in your "
                "environment or .env file."
            )
        self.client = groq.Groq(api_key=self.api_key)

    def ask(
        self,
        question: str,
        dm: DataManager,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        max_turns: Optional[int] = None,
    ) -> AgentResponse:
        max_turns = max_turns or settings.max_agent_turns

        # conversation_history holds prior user/assistant/tool turns (no system
        # message — that's rebuilt fresh each call so the schema context stays
        # current if new files were uploaded mid-conversation).
        messages: List[Dict[str, Any]] = list(conversation_history or [])
        messages.append({"role": "user", "content": question})

        system_message = {
            "role": "system",
            "content": SYSTEM_PROMPT.format(schema_context=dm.schema_context()),
        }
        steps: List[AgentStep] = []

        for turn in range(max_turns):
            try:
                print("=" * 60)
                print("MODEL:", self.model)
                print("MESSAGES:")
                print(json.dumps([system_message] + messages, indent=2, default=str))
                print("=" * 60)
                print("TOOLS:")
                print(json.dumps(TOOL_SCHEMAS, indent=2))
                print("=" * 60)

                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=settings.max_tokens,
                    messages=[system_message] + messages,
                    tools=TOOL_SCHEMAS,
                    tool_choice="auto",
                )
            except groq.APIError as e:
                logger.exception("Groq API error")
                return AgentResponse(
                    final_text=f"I hit an error calling the language model: {e}",
                    steps=steps,
                    raw_messages=messages,
                )

            choice = response.choices[0]
            msg = choice.message
            tool_calls = msg.tool_calls or []

            # Record the assistant turn (content may be None when it's tool-calls-only)
            assistant_msg: Dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
            if tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in tool_calls
                ]
            messages.append(assistant_msg)

            if not tool_calls:
                return AgentResponse(final_text=(msg.content or "").strip(), steps=steps, raw_messages=messages)

            for tc in tool_calls:
                try:
                    tool_input = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except json.JSONDecodeError:
                    tool_input = {}
                    logger.warning(f"Could not parse tool arguments: {tc.function.arguments!r}")

                logger.info(f"Turn {turn}: calling tool '{tc.function.name}' with input={tool_input}")
                result = execute_tool(tc.function.name, tool_input, dm)
                steps.append(AgentStep(tool_name=tc.function.name, tool_input=tool_input, tool_result=result))

                # Charts carry large figure JSON — don't replay that back into
                # the model's context, just confirm success so it can narrate.
                if tc.function.name == "create_chart" and result.get("success"):
                    llm_facing_result = {
                        "success": True,
                        "chart_created": True,
                        "spec": result.get("spec"),
                    }
                else:
                    llm_facing_result = result

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": safe_json(llm_facing_result),
                })

        return AgentResponse(
            final_text=(
                "I used up my available reasoning steps without reaching a final answer. "
                "Try rephrasing the question or breaking it into smaller parts."
            ),
            steps=steps,
            raw_messages=messages,
        )
