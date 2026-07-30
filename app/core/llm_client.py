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
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import groq

from app.config import settings
from app.core.data_manager import DataManager
from app.core.tools import TOOL_SCHEMAS, execute_tool, safe_json
from app.utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """
You are an AI data analyst.

Never invent values. Base every answer on tool results.

Rules:
- Use exactly one appropriate tool whenever possible.
- Use run_pandas_code for calculations and analysis.
- Use run_sql only if SQL is explicitly requested.
- Use create_chart only for visualizations.
- Use detect_anomalies only for anomaly or data-quality questions.
- Use get_dataset_info only if the schema below is insufficient.

For run_pandas_code:
- Datasets are already loaded as variables.
- pd, np, math, and statistics are already available.
- Never use import statements.
- Never read files.
- Always assign the final output to `result`.

If a tool returns enough information, answer immediately.
Do not call another tool unless information is missing.
Do not repeat the same tool call.

Schema:
{schema_context}
"""

MAX_TOOL_CALL_RETRIES = 1
REQUEST_TIMEOUT_S = 60
MAX_CONSECUTIVE_TOOL_FAILURES = 2

# --- Context-growth controls -------------------------------------------------
# Groq's on-demand tier for gpt-oss-20b caps at 8000 tokens/minute. Without
# bounding history, every past question's full tool-result payloads
# (dataframe previews, chart specs, etc.) accumulate in conversation_history
# and eventually blow past that limit on an unrelated later question
# (see: 413 rate_limit_exceeded in production logs).
MAX_HISTORY_MESSAGES = 12          # messages kept when starting a new ask()
TOOL_RESULT_HISTORY_CHARS = 300    # size cap for tool results once persisted

# --- Reasoning-token controls ------------------------------------------------
# gpt-oss models emit internal chain-of-thought before acting, which is
# counted against max_tokens and is the dominant cause of the 13-30s+ per-turn
# latency seen in the logs (vs <1s for calls that skip/short reasoning).
REASONING_EFFORT = settings.reasoning_effort  # "low" | "medium" | "high"
REASONING_FORMAT = "hidden"                   # don't return/pay to transmit the CoT


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
        self.client = groq.Groq(api_key=self.api_key, timeout=REQUEST_TIMEOUT_S)

        self._schema_cache_key: Optional[tuple] = None
        self._schema_cache_value: Optional[str] = None
        self._dataset_info_calls = 0

    # ------------------------------------------------------------------ #
    # Schema caching
    # ------------------------------------------------------------------ #
    def _cached_schema_context(self, dm: DataManager) -> str:
        cache_key = (id(dm), tuple(dm.frames.keys()))
        if cache_key != self._schema_cache_key:
            self._schema_cache_value = dm.schema_context()
            self._schema_cache_key = cache_key
        return self._schema_cache_value

    # ------------------------------------------------------------------ #
    # Context-size management
    # ------------------------------------------------------------------ #
    def _trim_history(self, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Keep only the most recent messages so a long-running session
        can't silently grow past the model's TPM limit. Avoids starting the
        trimmed window on a lone 'tool' message, which would break the
        assistant tool_calls -> tool role-message pairing the API expects."""
        if len(history) <= MAX_HISTORY_MESSAGES:
            return history
        trimmed = history[-MAX_HISTORY_MESSAGES:]
        while trimmed and trimmed[0].get("role") == "tool":
            trimmed = trimmed[1:]
        return trimmed

    def _compact_for_history(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Shrink tool-result payloads before persisting messages as
        conversation_history for a future ask() call. Full dataframe
        previews/chart specs only need to exist for the turn that produced
        them — keeping them forever is what caused requests to blow past
        the TPM cap on later, unrelated questions."""
        compacted = []
        for m in messages:
            if m.get("role") == "tool":
                m = dict(m)
                content = m.get("content", "")
                if len(content) > TOOL_RESULT_HISTORY_CHARS:
                    content = content[:TOOL_RESULT_HISTORY_CHARS] + "...[truncated for history]"
                m["content"] = content
            compacted.append(m)
        return compacted

    # ------------------------------------------------------------------ #
    # Groq call wrapper
    # ------------------------------------------------------------------ #
    def _is_tool_use_failed(self, e: groq.BadRequestError) -> bool:
        body = getattr(e, "body", None)
        if not isinstance(body, dict):
            return False
        err = body.get("error", {})
        return isinstance(err, dict) and err.get("code") == "tool_use_failed"

    def _create_completion(self, system_message: Dict[str, Any], messages: List[Dict[str, Any]]):
        """Call the Groq chat completions endpoint, retrying if the model
        emits a malformed tool call that Groq rejects server-side with a 400.
        Does NOT retry on 413/429 rate-limit errors — those need the caller
        to shrink the request, not resend the same oversized one."""
        last_error: Optional[Exception] = None

        extra_kwargs: Dict[str, Any] = {}
        if "gpt-oss" in self.model:
            # reasoning_effort/reasoning_format are only accepted for Groq's
            # gpt-oss reasoning models; sending them for other models (e.g.
            # llama3-*) is rejected with a 400.
            extra_kwargs["reasoning_effort"] = REASONING_EFFORT
            extra_kwargs["reasoning_format"] = REASONING_FORMAT

        for attempt in range(MAX_TOOL_CALL_RETRIES + 1):
            start = time.perf_counter()
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=settings.max_tokens,
                    messages=[system_message] + messages,
                    tools=TOOL_SCHEMAS,
                    tool_choice="auto",
                    **extra_kwargs,
                )
                logger.info(f"LLM call took {time.perf_counter() - start:.2f}s")
                return response
            except groq.BadRequestError as e:
                last_error = e
                if self._is_tool_use_failed(e) and attempt < MAX_TOOL_CALL_RETRIES:
                    failed_gen = None
                    body = getattr(e, "body", None)
                    if isinstance(body, dict):
                        failed_gen = body.get("error", {}).get("failed_generation")
                    logger.warning(
                        f"Groq emitted a malformed tool call (attempt {attempt + 1}/"
                        f"{MAX_TOOL_CALL_RETRIES + 1}), retrying: {failed_gen!r}"
                    )
                    continue
                raise
            except groq.APIStatusError as e:
                # 413 (request too large) / 429 (rate limited) — retrying the
                # identical request will just fail again. Surface immediately.
                last_error = e
                raise
            except groq.APIError as e:
                last_error = e
                raise

        raise last_error  # type: ignore[misc]

    # ------------------------------------------------------------------ #
    # Agent loop
    # ------------------------------------------------------------------ #
    def ask(
        self,
        question: str,
        dm: DataManager,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        max_turns: Optional[int] = None,
    ) -> AgentResponse:
        max_turns = max_turns or settings.max_agent_turns
        self._dataset_info_calls = 0
        consecutive_tool_failures = 0

        messages: List[Dict[str, Any]] = self._trim_history(list(conversation_history or []))
        messages.append({"role": "user", "content": question})

        system_message = {
            "role": "system",
            "content": SYSTEM_PROMPT.format(schema_context=self._cached_schema_context(dm)),
        }
        steps: List[AgentStep] = []

        for turn in range(max_turns):
            try:
                response = self._create_completion(system_message, messages)
                logger.info(f"Full response: {response}")
            except groq.APIStatusError as e:
                if e.status_code in (413, 429):
                    logger.warning(f"Request too large / rate limited: {e}")
                    return AgentResponse(
                        final_text=(
                            "This conversation has gotten too long for the model's rate "
                            "limit. Try asking again — I've trimmed the history — or start "
                            "a fresh conversation if it keeps happening."
                        ),
                        steps=steps,
                        raw_messages=self._compact_for_history(messages),
                    )
                logger.exception("Groq API error")
                return AgentResponse(
                    final_text=f"I hit an error calling the language model: {e}",
                    steps=steps,
                    raw_messages=self._compact_for_history(messages),
                )
            except groq.APIError as e:
                logger.exception("Groq API error")
                return AgentResponse(
                    final_text=f"I hit an error calling the language model: {e}",
                    steps=steps,
                    raw_messages=self._compact_for_history(messages),
                )
            
            try:
                choice = response.choices[0]
                msg = choice.message
                tool_calls = msg.tool_calls or []

                logger.info("=" * 60)
                logger.info(f"Assistant content: {msg.content}")
                logger.info(f"Tool calls: {tool_calls}")
                logger.info("=" * 60)

            except Exception:
                logger.exception("Failed while reading Groq response")
                raise

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
                return AgentResponse(
                    final_text=(msg.content or "").strip(),
                    steps=steps,
                    raw_messages=self._compact_for_history(messages),
                )

            for tc in tool_calls:
                try:
                    # Debug logging
                    logger.info("=" * 60)
                    logger.info(f"Tool Name: {tc.function.name}")
                    logger.info(f"Raw Arguments: {tc.function.arguments}")

                    tool_input = (
                        json.loads(tc.function.arguments)
                        if tc.function.arguments
                        else {}
                    )

                    logger.info(f"Parsed Arguments: {tool_input}")
                    logger.info("=" * 60)

                except json.JSONDecodeError:
                    tool_input = {}
                    logger.warning(
                        f"Could not parse tool arguments: {tc.function.arguments!r}"
                    )

                if tc.function.name == "get_dataset_info":
                    self._dataset_info_calls += 1
                    if self._dataset_info_calls > 1:
                        result = {
                            "success": True,
                            "note": "Already provided in the system prompt schema context above — use that instead of calling this again.",
                        }
                        steps.append(
                            AgentStep(tool_name=tc.function.name, tool_input=tool_input, tool_result=result)
                        )
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": safe_json(result),
                        })
                        consecutive_tool_failures = 0
                        continue

                logger.info(f"Turn {turn}: calling tool '{tc.function.name}' with input={tool_input}")
                tool_start = time.perf_counter()
                result = execute_tool(tc.function.name, tool_input, dm)
                logger.info(
                    f"Tool '{tc.function.name}' took {time.perf_counter() - tool_start:.2f}s "
                    f"(success={result.get('success')})"
                )
                steps.append(AgentStep(tool_name=tc.function.name, tool_input=tool_input, tool_result=result))

                if result.get("success"):
                    consecutive_tool_failures = 0
                else:
                    consecutive_tool_failures += 1

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

            if consecutive_tool_failures >= MAX_CONSECUTIVE_TOOL_FAILURES:
                return AgentResponse(
                    final_text=(
                        "I tried a couple of times but couldn't get a tool call to succeed "
                        "for this question — the last error was: "
                        f"{steps[-1].tool_result.get('error', 'unknown error')}. "
                        "Try rephrasing the question or checking the column names."
                    ),
                    steps=steps,
                    raw_messages=self._compact_for_history(messages),
                )

        return AgentResponse(
            final_text=(
                "I used up my available reasoning steps without reaching a final answer. "
                "Try rephrasing the question or breaking it into smaller parts."
            ),
            steps=steps,
            raw_messages=self._compact_for_history(messages),
        )