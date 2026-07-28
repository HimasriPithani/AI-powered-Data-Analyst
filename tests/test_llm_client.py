"""
Tests for LLMClient's agent loop against a mocked Groq client, so the
tool-call parsing / message-building logic is verified without needing a
real GROQ_API_KEY or network access.
"""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.core.data_manager import DataManager
from app.core.llm_client import LLMClient


def make_dm():
    dm = DataManager()
    dm.frames["sales"] = pd.DataFrame({"region": ["N", "S"], "revenue": [100, 200]})
    dm.profiles["sales"] = dm._profile("sales", dm.frames["sales"])
    return dm


def _mock_response(content=None, tool_calls=None):
    """Build a fake groq chat.completions.create() response object."""
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


def _mock_tool_call(call_id, name, arguments: dict):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


@patch("app.core.llm_client.groq.Groq")
def test_final_answer_with_no_tool_calls(mock_groq_cls):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_response(
        content="Total revenue is 300.", tool_calls=None
    )
    mock_groq_cls.return_value = mock_client

    client = LLMClient(api_key="fake-key")
    dm = make_dm()
    response = client.ask("What is total revenue?", dm)

    assert response.final_text == "Total revenue is 300."
    assert response.steps == []


@patch("app.core.llm_client.groq.Groq")
def test_tool_call_then_final_answer(mock_groq_cls):
    mock_client = MagicMock()
    tool_call = _mock_tool_call(
        "call_1", "run_pandas_code", {"code": "result = sales['revenue'].sum()"}
    )
    first_response = _mock_response(content=None, tool_calls=[tool_call])
    second_response = _mock_response(content="Revenue totals 300.", tool_calls=None)
    mock_client.chat.completions.create.side_effect = [first_response, second_response]
    mock_groq_cls.return_value = mock_client

    client = LLMClient(api_key="fake-key")
    dm = make_dm()
    response = client.ask("What is total revenue?", dm)

    assert response.final_text == "Revenue totals 300."
    assert len(response.steps) == 1
    assert response.steps[0].tool_name == "run_pandas_code"
    assert response.steps[0].tool_result["success"] is True
    assert response.steps[0].tool_result["result"] == 300

    # Check a tool message was appended to the conversation for the model to read
    tool_msgs = [m for m in response.raw_messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["tool_call_id"] == "call_1"


@patch("app.core.llm_client.groq.Groq")
def test_max_turns_budget_is_enforced(mock_groq_cls):
    mock_client = MagicMock()
    tool_call = _mock_tool_call("call_x", "get_dataset_info", {"dataset": "sales"})
    # Always returns another tool call, never a final answer
    mock_client.chat.completions.create.return_value = _mock_response(
        content=None, tool_calls=[tool_call]
    )
    mock_groq_cls.return_value = mock_client

    client = LLMClient(api_key="fake-key")
    dm = make_dm()
    response = client.ask("loop forever", dm, max_turns=2)

    assert "reasoning steps" in response.final_text.lower()
    assert len(response.steps) == 2
    assert mock_client.chat.completions.create.call_count == 2


def test_missing_api_key_raises():
    # No explicit key passed, and this test environment has no GROQ_API_KEY set,
    # so the client should refuse to initialize rather than silently proceeding.
    with pytest.raises(ValueError):
        LLMClient(api_key="")
