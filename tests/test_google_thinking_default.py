#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the low-latency thinking default on GoogleLLMService's out-of-band inference."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from google.genai.errors import ClientError

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.google.llm import GoogleLLMService


def _service(model: str) -> GoogleLLMService:
    return GoogleLLMService(api_key="test-key", settings=GoogleLLMService.Settings(model=model))


def _context() -> LLMContext:
    context = LLMContext()
    context.set_messages([{"role": "user", "content": "hello"}])
    return context


def _response(text: str = "ok"):
    return SimpleNamespace(
        usage_metadata=None,
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(text=text)]))],
    )


def _rejection(message: str) -> ClientError:
    return ClientError(
        400, {"error": {"code": 400, "message": message, "status": "INVALID_ARGUMENT"}}
    )


def _configs(mock: AsyncMock) -> list:
    return [call.kwargs["config"] for call in mock.await_args_list]


def test_default_thinking_per_model_family():
    """Gemini 2.5 Flash gets budget 0, Gemini 3 Flash gets level minimal, others nothing."""
    for model, expected in [
        ("gemini-2.5-flash", {"thinking_budget": 0}),
        ("gemini-2.5-flash-lite", {"thinking_budget": 0}),
        ("gemini-3.5-flash", {"thinking_level": "minimal"}),
        ("gemini-3.1-flash-lite", {"thinking_level": "minimal"}),
    ]:
        params: dict = {}
        assert _service(model)._maybe_unset_thinking_budget(params) is True, model
        assert params["thinking_config"] == expected, model

    params = {}
    assert _service("gemini-3-pro")._maybe_unset_thinking_budget(params) is False
    assert "thinking_config" not in params


def test_explicit_thinking_config_wins():
    """A configured thinking setting is never overridden by the default."""
    service = GoogleLLMService(
        api_key="test-key",
        settings=GoogleLLMService.Settings(
            model="gemini-3.5-flash",
            thinking=GoogleLLMService.ThinkingConfig(thinking_level="high"),
        ),
    )

    params = service._build_generation_params()
    assert service._maybe_unset_thinking_budget(params) is False
    assert params["thinking_config"] == {"thinking_level": "high"}


@pytest.mark.asyncio
async def test_run_inference_applies_thinking_default():
    """The out-of-band inference sends the same thinking default as the streaming path."""
    service = _service("gemini-3.5-flash")
    generate = AsyncMock(return_value=_response("answer"))
    service._client.aio.models.generate_content = generate

    text, _ = await service.run_inference_with_usage(_context())

    assert text == "answer"
    (config,) = _configs(generate)
    assert str(config.thinking_config.thinking_level.value).lower() == "minimal"


@pytest.mark.asyncio
async def test_run_inference_retries_without_default_when_model_rejects_it():
    """A 400 about thinking triggers one retry without the default; later calls skip it."""
    service = _service("gemini-3.5-flash")
    generate = AsyncMock(
        side_effect=[
            _rejection("thinking_level minimal is not supported for this model"),
            _response("first"),
            _response("second"),
        ]
    )
    service._client.aio.models.generate_content = generate

    first, _ = await service.run_inference_with_usage(_context())
    second, _ = await service.run_inference_with_usage(_context())

    assert (first, second) == ("first", "second")
    rejected, retried, later = _configs(generate)
    assert str(rejected.thinking_config.thinking_level.value).lower() == "minimal"
    assert retried.thinking_config is None
    assert later.thinking_config is None
    assert "gemini-3.5-flash" in service._thinking_default_rejected_models


@pytest.mark.asyncio
async def test_run_inference_propagates_unrelated_errors():
    """Only a thinking-config rejection is retried; other 400s surface unchanged."""
    service = _service("gemini-3.5-flash")
    generate = AsyncMock(side_effect=_rejection("Invalid JSON payload"))
    service._client.aio.models.generate_content = generate

    with pytest.raises(ClientError):
        await service.run_inference_with_usage(_context())

    assert generate.await_count == 1
    assert not service._thinking_default_rejected_models
