"""build_chat_completion_params: tool_choice must never outlive its tools.

A routing node parks tool_choice="required" in settings.extra, where it
persists across completions. A later request built with no tools (out-of-band
run_inference, or a node with an empty function list) must not carry the
orphan tool_choice — Google's OpenAI-compat endpoint rejects the whole request
with 400 "When using `tool_choice`, `tools` must be set".
"""

from openai import NOT_GIVEN

from pipecat.services.openai.llm import OpenAILLMService

TOOLS = [{"type": "function", "function": {"name": "f", "parameters": {}}}]


def _service() -> OpenAILLMService:
    service = OpenAILLMService(api_key="test-key")
    service._settings.extra["tool_choice"] = "required"
    return service


def test_tool_choice_dropped_when_tools_absent():
    params = _service().build_chat_completion_params(
        {"messages": [{"role": "user", "content": "hi"}]}
    )
    assert "tool_choice" not in params


def test_tool_choice_dropped_when_tools_not_given():
    # Context adapters pass tools=NOT_GIVEN when the node has no functions.
    params = _service().build_chat_completion_params(
        {"messages": [], "tools": NOT_GIVEN}
    )
    assert "tool_choice" not in params


def test_tool_choice_kept_when_tools_present():
    params = _service().build_chat_completion_params(
        {"messages": [], "tools": TOOLS}
    )
    assert params["tool_choice"] == "required"
    assert params["tools"] == TOOLS
