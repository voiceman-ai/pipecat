#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import NotGiven
from openai._types import NOT_GIVEN as OPENAI_NOT_GIVEN

from pipecat.adapters.services.anthropic_adapter import AnthropicLLMInvocationParams
from pipecat.adapters.services.bedrock_adapter import AWSBedrockLLMInvocationParams
from pipecat.adapters.services.gemini_adapter import GeminiLLMInvocationParams
from pipecat.adapters.services.open_ai_adapter import OpenAILLMInvocationParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.aws.llm import AWSBedrockLLMService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.llm_service import LLMService
from pipecat.services.openai.base_llm import INFERENCE_TIMEOUT_SECS
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.responses.llm import (
    OpenAIResponsesHttpLLMService,
    OpenAIResponsesLLMService,
)
from pipecat.services.openrouter.llm import OpenRouterLLMService


@pytest.mark.asyncio
async def test_openai_run_inference_with_llm_context():
    """Test run_inference with LLMContext returns expected response."""
    # Create service with mocked client and specific parameters
    with patch.object(OpenAILLMService, "create_client"):
        service = OpenAILLMService(
            settings=OpenAILLMService.Settings(
                model="gpt-4",
                temperature=0.7,
                max_tokens=100,
                frequency_penalty=0.5,
                seed=42,
            )
        )
        service._client = AsyncMock()

        # Setup mocks
        mock_context = MagicMock(spec=LLMContext)
        mock_adapter = MagicMock()
        test_messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello, world!"},
        ]
        mock_adapter.get_llm_invocation_params.return_value = OpenAILLMInvocationParams(
            messages=test_messages, tools=OPENAI_NOT_GIVEN, tool_choice=OPENAI_NOT_GIVEN
        )
        service.get_llm_adapter = MagicMock(return_value=mock_adapter)

        # Mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello! How can I help you today?"
        service._client.chat.completions.create.return_value = mock_response

        # Execute
        result = await service.run_inference(mock_context)

        # Verify
        assert result == "Hello! How can I help you today?"
        service.get_llm_adapter.assert_called_once()
        # convert_developer_to_user=False because OpenAILLMService.supports_developer_role is True
        mock_adapter.get_llm_invocation_params.assert_called_once_with(
            mock_context, system_instruction=None, convert_developer_to_user=False
        )
        service._client.chat.completions.create.assert_called_once_with(
            model="gpt-4",
            stream=False,
            frequency_penalty=0.5,
            presence_penalty=OPENAI_NOT_GIVEN,
            seed=42,
            temperature=0.7,
            top_p=OPENAI_NOT_GIVEN,
            max_tokens=100,
            max_completion_tokens=OPENAI_NOT_GIVEN,
            service_tier=OPENAI_NOT_GIVEN,
            messages=test_messages,
            tools=OPENAI_NOT_GIVEN,
            # No tools in this request, so tool_choice is dropped entirely —
            # some OpenAI-compatible endpoints 400 on an orphan tool_choice.
            # Out-of-band inference is non-streaming, so the client's
            # conversational read timeout would bound total generation rather
            # than time-to-first-byte. It carries its own, looser bound instead.
            timeout=INFERENCE_TIMEOUT_SECS,
        )


@pytest.mark.asyncio
async def test_openai_run_inference_client_exception():
    """Test that exceptions from the client are propagated."""
    with patch.object(OpenAILLMService, "create_client"):
        service = OpenAILLMService(settings=OpenAILLMService.Settings(model="gpt-4"))
        service._client = AsyncMock()

        mock_context = MagicMock(spec=LLMContext)
        mock_adapter = MagicMock()
        mock_adapter.get_llm_invocation_params.return_value = OpenAILLMInvocationParams(
            messages=[], tools=OPENAI_NOT_GIVEN, tool_choice=OPENAI_NOT_GIVEN
        )
        service.get_llm_adapter = MagicMock(return_value=mock_adapter)
        service._client.chat.completions.create.side_effect = Exception("API Error")

        with pytest.raises(Exception, match="API Error"):
            await service.run_inference(mock_context)


@pytest.mark.asyncio
async def test_openrouter_run_inference_converts_developer_messages_to_user():
    """Test OpenRouter requests convert developer messages for broad model compatibility."""
    with patch.object(OpenRouterLLMService, "create_client"):
        service = OpenRouterLLMService(settings=OpenRouterLLMService.Settings(model="gpt-4"))
        service._client = AsyncMock()

        mock_context = MagicMock(spec=LLMContext)
        mock_adapter = MagicMock()
        mock_adapter.get_llm_invocation_params.return_value = OpenAILLMInvocationParams(
            messages=[{"role": "user", "content": "Tool result"}],
            tools=OPENAI_NOT_GIVEN,
            tool_choice=OPENAI_NOT_GIVEN,
        )
        service.get_llm_adapter = MagicMock(return_value=mock_adapter)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Done"
        service._client.chat.completions.create.return_value = mock_response

        result = await service.run_inference(mock_context)

        assert result == "Done"
        mock_adapter.get_llm_invocation_params.assert_called_once_with(
            mock_context, system_instruction=None, convert_developer_to_user=True
        )


@pytest.mark.asyncio
async def test_anthropic_run_inference_with_llm_context():
    """Test run_inference with LLMContext returns expected response for Anthropic."""
    # Create service with mocked client and specific parameters
    from pipecat.services.anthropic.llm import AnthropicLLMService

    service = AnthropicLLMService(
        api_key="test-key",
        settings=AnthropicLLMService.Settings(
            model="claude-3-sonnet-20240229",
            max_tokens=2048,
            temperature=0.6,
            top_k=50,
            top_p=0.95,
        ),
    )
    service._client = AsyncMock()

    # Setup mocks
    mock_context = MagicMock(spec=LLMContext)
    mock_adapter = MagicMock()
    test_messages = [{"role": "user", "content": "Hello, world!"}]
    test_system = "You are a helpful assistant"
    mock_adapter.get_llm_invocation_params.return_value = AnthropicLLMInvocationParams(
        messages=test_messages, system=test_system, tools=[]
    )
    service.get_llm_adapter = MagicMock(return_value=mock_adapter)

    # Mock response
    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    mock_response.content[0].text = "Hello! How can I help you today?"
    service._client.beta.messages.create.return_value = mock_response

    # Execute
    result = await service.run_inference(mock_context)

    # Verify
    assert result == "Hello! How can I help you today?"
    service.get_llm_adapter.assert_called_once()
    mock_adapter.get_llm_invocation_params.assert_called_once_with(
        mock_context,
        enable_prompt_caching=False,
        system_instruction=None,
        ensure_last_message_is_user=False,
    )
    service._client.beta.messages.create.assert_called_once_with(
        model="claude-3-sonnet-20240229",
        max_tokens=2048,
        stream=False,
        temperature=0.6,
        top_k=50,
        top_p=0.95,
        messages=test_messages,
        system=test_system,
        tools=[],
        betas=["interleaved-thinking-2025-05-14"],
    )


@pytest.mark.asyncio
async def test_anthropic_run_inference_client_exception():
    """Test that exceptions from the Anthropic client are propagated."""
    service = AnthropicLLMService(
        api_key="test-key", settings=AnthropicLLMService.Settings(model="claude-3-sonnet-20240229")
    )
    service._client = AsyncMock()

    mock_context = MagicMock(spec=LLMContext)
    mock_adapter = MagicMock()
    mock_adapter.get_llm_invocation_params.return_value = AnthropicLLMInvocationParams(
        messages=[], system="Test system", tools=[]
    )
    service.get_llm_adapter = MagicMock(return_value=mock_adapter)
    service._client.beta.messages.create.side_effect = Exception("Anthropic API Error")

    with pytest.raises(Exception, match="Anthropic API Error"):
        await service.run_inference(mock_context)


@pytest.mark.asyncio
async def test_google_run_inference_with_llm_context():
    """Test run_inference with LLMContext returns expected response for Google."""
    # Create service with mocked client
    service = GoogleLLMService(
        api_key="test-key", settings=GoogleLLMService.Settings(model="gemini-2.0-flash")
    )
    service._client = AsyncMock()

    # Setup mocks
    mock_context = MagicMock(spec=LLMContext)
    mock_adapter = MagicMock()
    test_messages = [{"role": "user", "content": "Hello, world!"}]
    test_system = "You are a helpful assistant"
    mock_adapter.get_llm_invocation_params.return_value = GeminiLLMInvocationParams(
        messages=test_messages, system_instruction=test_system, tools=NotGiven()
    )
    service.get_llm_adapter = MagicMock(return_value=mock_adapter)

    # Mock response
    mock_response = MagicMock()
    mock_response.candidates = [MagicMock()]
    mock_response.candidates[0].content = MagicMock()
    mock_response.candidates[0].content.parts = [MagicMock()]
    mock_response.candidates[0].content.parts[0].text = "Hello! How can I help you today?"
    service._client.aio = AsyncMock()
    service._client.aio.models = AsyncMock()
    service._client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    # Execute
    result = await service.run_inference(mock_context)

    # Verify
    assert result == "Hello! How can I help you today?"
    service.get_llm_adapter.assert_called_once()
    mock_adapter.get_llm_invocation_params.assert_called_once_with(
        mock_context, system_instruction=None
    )
    service._client.aio.models.generate_content.assert_called_once()


@pytest.mark.asyncio
async def test_google_run_inference_client_exception():
    """Test that exceptions from the Google client are propagated."""
    service = GoogleLLMService(
        api_key="test-key", settings=GoogleLLMService.Settings(model="gemini-2.0-flash")
    )
    service._client = AsyncMock()

    mock_context = MagicMock(spec=LLMContext)
    mock_adapter = MagicMock()
    mock_adapter.get_llm_invocation_params.return_value = GeminiLLMInvocationParams(
        messages=[], system_instruction="Test system", tools=NotGiven()
    )
    service.get_llm_adapter = MagicMock(return_value=mock_adapter)
    service._client.aio = AsyncMock()
    service._client.aio.models = AsyncMock()
    service._client.aio.models.generate_content = AsyncMock(
        side_effect=Exception("Google API Error")
    )

    with pytest.raises(Exception, match="Google API Error"):
        await service.run_inference(mock_context)


@pytest.mark.asyncio
async def test_aws_bedrock_run_inference_with_llm_context():
    """Test run_inference with LLMContext returns expected response for AWS Bedrock."""
    # Create service with specific parameters
    from pipecat.services.aws.llm import AWSBedrockLLMService

    service = AWSBedrockLLMService(
        settings=AWSBedrockLLMService.Settings(
            model="anthropic.claude-3-sonnet-20240229-v1:0",
            max_tokens=1024,
            temperature=0.5,
            top_p=0.85,
        )
    )

    # Setup mocks
    mock_context = MagicMock(spec=LLMContext)
    mock_adapter = MagicMock()
    test_messages = [{"role": "user", "content": [{"text": "Hello, world!"}]}]
    test_system = [{"text": "You are a helpful assistant"}]
    mock_adapter.get_llm_invocation_params.return_value = AWSBedrockLLMInvocationParams(
        messages=test_messages, system=test_system, tools=[], tool_choice=None
    )
    service.get_llm_adapter = MagicMock(return_value=mock_adapter)

    # Mock the client and response
    mock_client = AsyncMock()
    mock_response = {
        "output": {"message": {"content": [{"text": "Hello! How can I help you today?"}]}}
    }
    mock_client.converse.return_value = mock_response

    # Patch the _aws_session.create_client method to be an async context manager
    mock_context_manager = AsyncMock()
    mock_context_manager.__aenter__ = AsyncMock(return_value=mock_client)
    mock_context_manager.__aexit__ = AsyncMock(return_value=None)

    with patch.object(service._aws_session, "create_client", return_value=mock_context_manager):
        # Execute
        result = await service.run_inference(mock_context)

        # Verify
        assert result == "Hello! How can I help you today?"
        service.get_llm_adapter.assert_called_once()
        mock_adapter.get_llm_invocation_params.assert_called_once_with(
            mock_context, system_instruction=None, ensure_last_message_is_user=False
        )

        # Verify the call includes configured parameters
        call_kwargs = mock_client.converse.call_args.kwargs
        assert call_kwargs["modelId"] == "anthropic.claude-3-sonnet-20240229-v1:0"
        assert call_kwargs["messages"] == test_messages
        assert call_kwargs["system"] == test_system
        assert call_kwargs["additionalModelRequestFields"] == {}
        assert "inferenceConfig" in call_kwargs
        assert call_kwargs["inferenceConfig"]["maxTokens"] == 1024
        assert call_kwargs["inferenceConfig"]["temperature"] == 0.5
        assert call_kwargs["inferenceConfig"]["topP"] == 0.85


@pytest.mark.asyncio
async def test_aws_bedrock_run_inference_client_exception():
    """Test that exceptions from the AWS Bedrock client are propagated."""
    service = AWSBedrockLLMService(
        settings=AWSBedrockLLMService.Settings(model="anthropic.claude-3-sonnet-20240229-v1:0")
    )

    mock_context = MagicMock(spec=LLMContext)
    mock_adapter = MagicMock()
    mock_adapter.get_llm_invocation_params.return_value = AWSBedrockLLMInvocationParams(
        messages=[], system=[{"text": "Test system"}], tools=[], tool_choice=None
    )
    service.get_llm_adapter = MagicMock(return_value=mock_adapter)

    # Mock AWS client to raise exception
    mock_client = AsyncMock()
    mock_client.converse.side_effect = Exception("Bedrock API Error")

    # Patch the _aws_session.create_client method to be an async context manager
    mock_context_manager = AsyncMock()
    mock_context_manager.__aenter__ = AsyncMock(return_value=mock_client)
    mock_context_manager.__aexit__ = AsyncMock(return_value=None)

    with patch.object(service._aws_session, "create_client", return_value=mock_context_manager):
        with pytest.raises(Exception, match="Bedrock API Error"):
            await service.run_inference(mock_context)


@pytest.mark.asyncio
async def test_aws_bedrock_streaming_captures_all_tool_calls():
    """Test AWS Bedrock streaming captures every parallel tool call, not just the last."""
    service = AWSBedrockLLMService(
        settings=AWSBedrockLLMService.Settings(model="anthropic.claude-3-sonnet-20240229-v1:0")
    )

    mock_context = MagicMock(spec=LLMContext)
    mock_adapter = MagicMock()
    mock_adapter.get_llm_invocation_params.return_value = AWSBedrockLLMInvocationParams(
        messages=[{"role": "user", "content": [{"text": "What's the weather?"}]}],
        system=[],
        tools=[{"toolSpec": {"name": "get_weather"}}],
        tool_choice=None,
    )
    mock_adapter.get_messages_for_logging.return_value = []
    service.get_llm_adapter = MagicMock(return_value=mock_adapter)

    # Two tool calls in one streamed response, each its own content block keyed
    # by contentBlockIndex, finalized with contentBlockStop before messageStop.
    stream_events = [
        {
            "contentBlockStart": {
                "contentBlockIndex": 0,
                "start": {"toolUse": {"toolUseId": "id-0", "name": "get_weather"}},
            }
        },
        {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"toolUse": {"input": '{"city": "SF"}'}},
            }
        },
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {
            "contentBlockStart": {
                "contentBlockIndex": 1,
                "start": {"toolUse": {"toolUseId": "id-1", "name": "get_weather"}},
            }
        },
        {
            "contentBlockDelta": {
                "contentBlockIndex": 1,
                "delta": {"toolUse": {"input": '{"city": "NY"}'}},
            }
        },
        {"contentBlockStop": {"contentBlockIndex": 1}},
        {"messageStop": {"stopReason": "tool_use"}},
    ]

    async def fake_stream():
        for event in stream_events:
            yield event

    mock_client = AsyncMock()
    mock_context_manager = AsyncMock()
    mock_context_manager.__aenter__ = AsyncMock(return_value=mock_client)
    mock_context_manager.__aexit__ = AsyncMock(return_value=None)

    captured = []
    service.run_function_calls = AsyncMock(side_effect=lambda calls: captured.extend(calls))

    with (
        patch.object(service._aws_session, "create_client", return_value=mock_context_manager),
        patch.object(service, "_create_converse_stream", return_value={"stream": fake_stream()}),
    ):
        await service._process_context(mock_context)

    assert len(captured) == 2
    assert sorted(c.arguments["city"] for c in captured) == ["NY", "SF"]
    assert sorted(c.tool_call_id for c in captured) == ["id-0", "id-1"]


# --- system_instruction parameter tests ---


@pytest.mark.asyncio
async def test_openai_run_inference_system_instruction_overrides_context():
    """Test that system_instruction overrides the system message from context."""
    with patch.object(OpenAILLMService, "create_client"):
        service = OpenAILLMService(settings=OpenAILLMService.Settings(model="gpt-4"))
        service._client = AsyncMock()

        mock_context = MagicMock(spec=LLMContext)
        mock_adapter = MagicMock()
        test_messages = [
            {"role": "system", "content": "Original system message"},
            {"role": "user", "content": "Hello"},
        ]
        mock_adapter.get_llm_invocation_params.return_value = OpenAILLMInvocationParams(
            messages=test_messages, tools=OPENAI_NOT_GIVEN, tool_choice=OPENAI_NOT_GIVEN
        )
        service.get_llm_adapter = MagicMock(return_value=mock_adapter)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        service._client.chat.completions.create.return_value = mock_response

        result = await service.run_inference(
            mock_context, system_instruction="New system instruction"
        )

        assert result == "Response"
        # Verify the adapter was called with the correct system_instruction.
        # convert_developer_to_user=False because OpenAILLMService.supports_developer_role is True.
        mock_adapter.get_llm_invocation_params.assert_called_once_with(
            mock_context,
            system_instruction="New system instruction",
            convert_developer_to_user=False,
        )


@pytest.mark.asyncio
async def test_openai_run_inference_system_instruction_none_unchanged():
    """Test that when system_instruction is None, behavior is unchanged."""
    with patch.object(OpenAILLMService, "create_client"):
        service = OpenAILLMService(settings=OpenAILLMService.Settings(model="gpt-4"))
        service._client = AsyncMock()

        mock_context = MagicMock(spec=LLMContext)
        mock_adapter = MagicMock()
        test_messages = [
            {"role": "system", "content": "Original system message"},
            {"role": "user", "content": "Hello"},
        ]
        mock_adapter.get_llm_invocation_params.return_value = OpenAILLMInvocationParams(
            messages=test_messages, tools=OPENAI_NOT_GIVEN, tool_choice=OPENAI_NOT_GIVEN
        )
        service.get_llm_adapter = MagicMock(return_value=mock_adapter)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        service._client.chat.completions.create.return_value = mock_response

        result = await service.run_inference(mock_context)

        assert result == "Response"
        call_kwargs = service._client.chat.completions.create.call_args.kwargs
        messages = call_kwargs["messages"]
        assert messages[0] == {"role": "system", "content": "Original system message"}
        assert messages[1] == {"role": "user", "content": "Hello"}


@pytest.mark.asyncio
async def test_anthropic_run_inference_system_instruction_overrides_context():
    """Test that system_instruction overrides the system message for Anthropic."""
    service = AnthropicLLMService(
        api_key="test-key", settings=AnthropicLLMService.Settings(model="claude-3-sonnet-20240229")
    )
    service._client = AsyncMock()

    mock_context = MagicMock(spec=LLMContext)
    mock_adapter = MagicMock()
    test_messages = [{"role": "user", "content": "Hello"}]
    mock_adapter.get_llm_invocation_params.return_value = AnthropicLLMInvocationParams(
        messages=test_messages, system="Original system", tools=[]
    )
    service.get_llm_adapter = MagicMock(return_value=mock_adapter)

    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    mock_response.content[0].text = "Response"
    service._client.beta.messages.create.return_value = mock_response

    result = await service.run_inference(mock_context, system_instruction="New system instruction")

    assert result == "Response"
    # Verify the adapter was called with the correct system_instruction
    mock_adapter.get_llm_invocation_params.assert_called_once_with(
        mock_context,
        enable_prompt_caching=False,
        system_instruction="New system instruction",
        ensure_last_message_is_user=False,
    )


@pytest.mark.asyncio
async def test_anthropic_run_inference_system_instruction_none_unchanged():
    """Test that when system_instruction is None, Anthropic behavior is unchanged."""
    service = AnthropicLLMService(
        api_key="test-key", settings=AnthropicLLMService.Settings(model="claude-3-sonnet-20240229")
    )
    service._client = AsyncMock()

    mock_context = MagicMock(spec=LLMContext)
    mock_adapter = MagicMock()
    test_messages = [{"role": "user", "content": "Hello"}]
    mock_adapter.get_llm_invocation_params.return_value = AnthropicLLMInvocationParams(
        messages=test_messages, system="Original system", tools=[]
    )
    service.get_llm_adapter = MagicMock(return_value=mock_adapter)

    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    mock_response.content[0].text = "Response"
    service._client.beta.messages.create.return_value = mock_response

    result = await service.run_inference(mock_context)

    assert result == "Response"
    call_kwargs = service._client.beta.messages.create.call_args.kwargs
    assert call_kwargs["system"] == "Original system"


@pytest.mark.asyncio
async def test_google_run_inference_system_instruction_overrides_context():
    """Test that system_instruction overrides the system message for Google."""
    service = GoogleLLMService(
        api_key="test-key", settings=GoogleLLMService.Settings(model="gemini-2.0-flash")
    )
    service._client = AsyncMock()

    mock_context = MagicMock(spec=LLMContext)
    mock_adapter = MagicMock()
    test_messages = [{"role": "user", "content": "Hello"}]
    mock_adapter.get_llm_invocation_params.return_value = GeminiLLMInvocationParams(
        messages=test_messages, system_instruction="Original system", tools=NotGiven()
    )
    service.get_llm_adapter = MagicMock(return_value=mock_adapter)

    mock_response = MagicMock()
    mock_response.candidates = [MagicMock()]
    mock_response.candidates[0].content = MagicMock()
    mock_response.candidates[0].content.parts = [MagicMock()]
    mock_response.candidates[0].content.parts[0].text = "Response"
    service._client.aio = AsyncMock()
    service._client.aio.models = AsyncMock()
    service._client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    result = await service.run_inference(mock_context, system_instruction="New system instruction")

    assert result == "Response"
    # Verify the adapter was called with the correct system_instruction
    mock_adapter.get_llm_invocation_params.assert_called_once_with(
        mock_context, system_instruction="New system instruction"
    )


@pytest.mark.asyncio
async def test_google_run_inference_system_instruction_none_unchanged():
    """Test that when system_instruction is None, Google behavior is unchanged."""
    service = GoogleLLMService(
        api_key="test-key", settings=GoogleLLMService.Settings(model="gemini-2.0-flash")
    )
    service._client = AsyncMock()

    mock_context = MagicMock(spec=LLMContext)
    mock_adapter = MagicMock()
    test_messages = [{"role": "user", "content": "Hello"}]
    mock_adapter.get_llm_invocation_params.return_value = GeminiLLMInvocationParams(
        messages=test_messages, system_instruction="Original system", tools=NotGiven()
    )
    service.get_llm_adapter = MagicMock(return_value=mock_adapter)

    mock_response = MagicMock()
    mock_response.candidates = [MagicMock()]
    mock_response.candidates[0].content = MagicMock()
    mock_response.candidates[0].content.parts = [MagicMock()]
    mock_response.candidates[0].content.parts[0].text = "Response"
    service._client.aio = AsyncMock()
    service._client.aio.models = AsyncMock()
    service._client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    result = await service.run_inference(mock_context)

    assert result == "Response"
    call_kwargs = service._client.aio.models.generate_content.call_args.kwargs
    config = call_kwargs["config"]
    assert config.system_instruction == "Original system"


@pytest.mark.asyncio
async def test_aws_bedrock_run_inference_system_instruction_overrides_context():
    """Test that system_instruction overrides the system message for AWS Bedrock."""
    service = AWSBedrockLLMService(
        settings=AWSBedrockLLMService.Settings(model="anthropic.claude-3-sonnet-20240229-v1:0")
    )

    mock_context = MagicMock(spec=LLMContext)
    mock_adapter = MagicMock()
    test_messages = [{"role": "user", "content": [{"text": "Hello"}]}]
    mock_adapter.get_llm_invocation_params.return_value = AWSBedrockLLMInvocationParams(
        messages=test_messages,
        system=[{"text": "Original system"}],
        tools=[],
        tool_choice=None,
    )
    service.get_llm_adapter = MagicMock(return_value=mock_adapter)

    mock_client = AsyncMock()
    mock_response = {"output": {"message": {"content": [{"text": "Response"}]}}}
    mock_client.converse.return_value = mock_response

    mock_context_manager = AsyncMock()
    mock_context_manager.__aenter__ = AsyncMock(return_value=mock_client)
    mock_context_manager.__aexit__ = AsyncMock(return_value=None)

    with patch.object(service._aws_session, "create_client", return_value=mock_context_manager):
        result = await service.run_inference(
            mock_context, system_instruction="New system instruction"
        )

        assert result == "Response"
        # Verify the adapter was called with the correct system_instruction
        mock_adapter.get_llm_invocation_params.assert_called_once_with(
            mock_context,
            system_instruction="New system instruction",
            ensure_last_message_is_user=False,
        )


@pytest.mark.asyncio
async def test_aws_bedrock_run_inference_system_instruction_none_unchanged():
    """Test that when system_instruction is None, AWS Bedrock behavior is unchanged."""
    service = AWSBedrockLLMService(
        settings=AWSBedrockLLMService.Settings(model="anthropic.claude-3-sonnet-20240229-v1:0")
    )

    mock_context = MagicMock(spec=LLMContext)
    mock_adapter = MagicMock()
    test_messages = [{"role": "user", "content": [{"text": "Hello"}]}]
    mock_adapter.get_llm_invocation_params.return_value = AWSBedrockLLMInvocationParams(
        messages=test_messages,
        system=[{"text": "Original system"}],
        tools=[],
        tool_choice=None,
    )
    service.get_llm_adapter = MagicMock(return_value=mock_adapter)

    mock_client = AsyncMock()
    mock_response = {"output": {"message": {"content": [{"text": "Response"}]}}}
    mock_client.converse.return_value = mock_response

    mock_context_manager = AsyncMock()
    mock_context_manager.__aenter__ = AsyncMock(return_value=mock_client)
    mock_context_manager.__aexit__ = AsyncMock(return_value=None)

    with patch.object(service._aws_session, "create_client", return_value=mock_context_manager):
        result = await service.run_inference(mock_context)

        assert result == "Response"
        call_kwargs = mock_client.converse.call_args.kwargs
        assert call_kwargs["system"] == [{"text": "Original system"}]


# --- OpenAI Responses API tests ---


@pytest.mark.asyncio
async def test_openai_responses_run_inference_with_llm_context():
    """Test run_inference with LLMContext returns expected response."""
    with patch.object(OpenAIResponsesLLMService, "_create_client"):
        service = OpenAIResponsesLLMService(
            settings=OpenAIResponsesLLMService.Settings(
                model="gpt-4.1",
                system_instruction="You are a helpful assistant",
                temperature=0.7,
                max_completion_tokens=100,
            ),
        )
        service._client = AsyncMock()

        context = LLMContext(
            messages=[
                {"role": "user", "content": "Hello, world!"},
            ]
        )

        mock_response = MagicMock()
        mock_response.output_text = "Hello! How can I help you today?"
        service._client.responses.create = AsyncMock(return_value=mock_response)

        result = await service.run_inference(context)

        assert result == "Hello! How can I help you today?"
        call_kwargs = service._client.responses.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4.1"
        assert call_kwargs["stream"] is False
        assert call_kwargs["store"] is False
        assert call_kwargs["input"] == [{"role": "user", "content": "Hello, world!"}]
        assert call_kwargs["instructions"] == "You are a helpful assistant"
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_output_tokens"] == 100


@pytest.mark.asyncio
async def test_openai_responses_run_inference_client_exception():
    """Test that exceptions from the client are propagated."""
    with patch.object(OpenAIResponsesLLMService, "_create_client"):
        service = OpenAIResponsesLLMService()
        service._client = AsyncMock()

        context = LLMContext(messages=[{"role": "user", "content": "Hello"}])
        service._client.responses.create = AsyncMock(side_effect=Exception("API Error"))

        with pytest.raises(Exception, match="API Error"):
            await service.run_inference(context)


@pytest.mark.asyncio
async def test_openai_responses_run_inference_system_instruction_overrides():
    """Test that system_instruction parameter overrides the settings instruction."""
    with patch.object(OpenAIResponsesLLMService, "_create_client"):
        service = OpenAIResponsesLLMService(
            settings=OpenAIResponsesLLMService.Settings(
                model="gpt-4.1",
                system_instruction="Original instruction",
            ),
        )
        service._client = AsyncMock()

        context = LLMContext(
            messages=[{"role": "user", "content": "Hello"}],
        )

        mock_response = MagicMock()
        mock_response.output_text = "Response"
        service._client.responses.create = AsyncMock(return_value=mock_response)

        result = await service.run_inference(context, system_instruction="New system instruction")

        assert result == "Response"
        call_kwargs = service._client.responses.create.call_args.kwargs
        assert call_kwargs["instructions"] == "New system instruction"
        assert call_kwargs["input"] == [{"role": "user", "content": "Hello"}]


@pytest.mark.asyncio
async def test_openai_responses_run_inference_empty_context_with_instruction():
    """Test that system_instruction becomes a developer message when context is empty."""
    with patch.object(OpenAIResponsesLLMService, "_create_client"):
        service = OpenAIResponsesLLMService(
            settings=OpenAIResponsesLLMService.Settings(
                model="gpt-4.1",
                system_instruction="You are helpful",
            ),
        )
        service._client = AsyncMock()

        context = LLMContext(messages=[])

        mock_response = MagicMock()
        mock_response.output_text = "Response"
        service._client.responses.create = AsyncMock(return_value=mock_response)

        result = await service.run_inference(context)

        assert result == "Response"
        call_kwargs = service._client.responses.create.call_args.kwargs
        # With empty context, instruction should become a developer message
        assert call_kwargs["input"] == [{"role": "developer", "content": "You are helpful"}]
        assert "instructions" not in call_kwargs


@pytest.mark.asyncio
async def test_openai_responses_run_inference_max_tokens_override():
    """Test that max_tokens parameter overrides max_output_tokens."""
    with patch.object(OpenAIResponsesLLMService, "_create_client"):
        service = OpenAIResponsesLLMService(
            settings=OpenAIResponsesLLMService.Settings(
                model="gpt-4.1",
                max_completion_tokens=500,
            ),
        )
        service._client = AsyncMock()

        context = LLMContext(
            messages=[{"role": "user", "content": "Summarize this"}],
        )

        mock_response = MagicMock()
        mock_response.output_text = "Summary"
        service._client.responses.create = AsyncMock(return_value=mock_response)

        result = await service.run_inference(context, max_tokens=200)

        assert result == "Summary"
        call_kwargs = service._client.responses.create.call_args.kwargs
        assert call_kwargs["max_output_tokens"] == 200


@pytest.mark.asyncio
async def test_openai_responses_run_inference_system_instruction_param_with_empty_context():
    """Test that system_instruction param becomes a developer message when context is empty.

    The Responses API rejects requests with instructions but no input items.
    When run_inference is called with an explicit system_instruction and an
    empty context, the instruction must become a developer message — not be
    sent as the instructions parameter.
    """
    with patch.object(OpenAIResponsesLLMService, "_create_client"):
        service = OpenAIResponsesLLMService(
            settings=OpenAIResponsesLLMService.Settings(model="gpt-4.1"),
        )
        service._client = AsyncMock()

        context = LLMContext(messages=[])

        mock_response = MagicMock()
        mock_response.output_text = "Response"
        service._client.responses.create = AsyncMock(return_value=mock_response)

        result = await service.run_inference(
            context, system_instruction="Summarize the conversation"
        )

        assert result == "Response"
        call_kwargs = service._client.responses.create.call_args.kwargs
        assert call_kwargs["input"] == [
            {"role": "developer", "content": "Summarize the conversation"}
        ]
        assert "instructions" not in call_kwargs


# --- OpenAI Responses HTTP API tests ---
# These mirror the WebSocket variant tests above, verifying that the HTTP
# variant's run_inference (inherited from the shared base class) works
# identically.


@pytest.mark.asyncio
async def test_openai_responses_http_run_inference_with_llm_context():
    """Test run_inference with LLMContext returns expected response (HTTP variant)."""
    with patch.object(OpenAIResponsesHttpLLMService, "_create_client"):
        service = OpenAIResponsesHttpLLMService(
            settings=OpenAIResponsesHttpLLMService.Settings(
                model="gpt-4.1",
                system_instruction="You are a helpful assistant",
                temperature=0.7,
                max_completion_tokens=100,
            ),
        )
        service._client = AsyncMock()

        context = LLMContext(
            messages=[
                {"role": "user", "content": "Hello, world!"},
            ]
        )

        mock_response = MagicMock()
        mock_response.output_text = "Hello! How can I help you today?"
        service._client.responses.create = AsyncMock(return_value=mock_response)

        result = await service.run_inference(context)

        assert result == "Hello! How can I help you today?"
        call_kwargs = service._client.responses.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4.1"
        assert call_kwargs["stream"] is False
        assert call_kwargs["store"] is False
        assert call_kwargs["input"] == [{"role": "user", "content": "Hello, world!"}]
        assert call_kwargs["instructions"] == "You are a helpful assistant"
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_output_tokens"] == 100


@pytest.mark.asyncio
async def test_openai_responses_http_run_inference_client_exception():
    """Test that exceptions from the client are propagated (HTTP variant)."""
    with patch.object(OpenAIResponsesHttpLLMService, "_create_client"):
        service = OpenAIResponsesHttpLLMService()
        service._client = AsyncMock()

        context = LLMContext(messages=[{"role": "user", "content": "Hello"}])
        service._client.responses.create = AsyncMock(side_effect=Exception("API Error"))

        with pytest.raises(Exception, match="API Error"):
            await service.run_inference(context)


@pytest.mark.asyncio
async def test_openai_responses_http_run_inference_system_instruction_overrides():
    """Test that system_instruction parameter overrides the settings instruction (HTTP variant)."""
    with patch.object(OpenAIResponsesHttpLLMService, "_create_client"):
        service = OpenAIResponsesHttpLLMService(
            settings=OpenAIResponsesHttpLLMService.Settings(
                model="gpt-4.1",
                system_instruction="Original instruction",
            ),
        )
        service._client = AsyncMock()

        context = LLMContext(
            messages=[{"role": "user", "content": "Hello"}],
        )

        mock_response = MagicMock()
        mock_response.output_text = "Response"
        service._client.responses.create = AsyncMock(return_value=mock_response)

        result = await service.run_inference(context, system_instruction="New system instruction")

        assert result == "Response"
        call_kwargs = service._client.responses.create.call_args.kwargs
        assert call_kwargs["instructions"] == "New system instruction"
        assert call_kwargs["input"] == [{"role": "user", "content": "Hello"}]


@pytest.mark.asyncio
async def test_openai_responses_http_run_inference_empty_context_with_instruction():
    """Test that system_instruction becomes a developer message when context is empty (HTTP)."""
    with patch.object(OpenAIResponsesHttpLLMService, "_create_client"):
        service = OpenAIResponsesHttpLLMService(
            settings=OpenAIResponsesHttpLLMService.Settings(
                model="gpt-4.1",
                system_instruction="You are helpful",
            ),
        )
        service._client = AsyncMock()

        context = LLMContext(messages=[])

        mock_response = MagicMock()
        mock_response.output_text = "Response"
        service._client.responses.create = AsyncMock(return_value=mock_response)

        result = await service.run_inference(context)

        assert result == "Response"
        call_kwargs = service._client.responses.create.call_args.kwargs
        assert call_kwargs["input"] == [{"role": "developer", "content": "You are helpful"}]
        assert "instructions" not in call_kwargs


@pytest.mark.asyncio
async def test_openai_responses_http_run_inference_max_tokens_override():
    """Test that max_tokens parameter overrides max_output_tokens (HTTP variant)."""
    with patch.object(OpenAIResponsesHttpLLMService, "_create_client"):
        service = OpenAIResponsesHttpLLMService(
            settings=OpenAIResponsesHttpLLMService.Settings(
                model="gpt-4.1",
                max_completion_tokens=500,
            ),
        )
        service._client = AsyncMock()

        context = LLMContext(
            messages=[{"role": "user", "content": "Summarize this"}],
        )

        mock_response = MagicMock()
        mock_response.output_text = "Summary"
        service._client.responses.create = AsyncMock(return_value=mock_response)

        result = await service.run_inference(context, max_tokens=200)

        assert result == "Summary"
        call_kwargs = service._client.responses.create.call_args.kwargs
        assert call_kwargs["max_output_tokens"] == 200


@pytest.mark.asyncio
async def test_openai_responses_http_run_inference_system_instruction_param_with_empty_context():
    """Test system_instruction param becomes developer message for empty context (HTTP)."""
    with patch.object(OpenAIResponsesHttpLLMService, "_create_client"):
        service = OpenAIResponsesHttpLLMService(
            settings=OpenAIResponsesHttpLLMService.Settings(model="gpt-4.1"),
        )
        service._client = AsyncMock()

        context = LLMContext(messages=[])

        mock_response = MagicMock()
        mock_response.output_text = "Response"
        service._client.responses.create = AsyncMock(return_value=mock_response)

        result = await service.run_inference(
            context, system_instruction="Summarize the conversation"
        )

        assert result == "Response"
        call_kwargs = service._client.responses.create.call_args.kwargs
        assert call_kwargs["input"] == [
            {"role": "developer", "content": "Summarize the conversation"}
        ]
        assert "instructions" not in call_kwargs


# --- run_inference_with_usage: the usage-reporting seam -----------------------
#
# Every provider must report prompt_tokens INCLUSIVE of cache read/creation
# tokens (see LLMService.run_inference_with_usage), so consumers can price any
# provider's usage with the same arithmetic. These tests pin the per-provider
# normalization.


def _mocked_openai_service():
    with patch.object(OpenAILLMService, "create_client"):
        service = OpenAILLMService(settings=OpenAILLMService.Settings(model="gpt-4"))
    service._client = AsyncMock()
    mock_adapter = MagicMock()
    mock_adapter.get_llm_invocation_params.return_value = OpenAILLMInvocationParams(
        messages=[], tools=OPENAI_NOT_GIVEN, tool_choice=OPENAI_NOT_GIVEN
    )
    service.get_llm_adapter = MagicMock(return_value=mock_adapter)
    return service


@pytest.mark.asyncio
async def test_openai_run_inference_with_usage():
    """Chat-completions usage passes through (prompt already cache-inclusive)."""
    service = _mocked_openai_service()

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "hi"
    mock_response.usage = SimpleNamespace(
        prompt_tokens=1000,
        completion_tokens=50,
        total_tokens=1050,
        prompt_tokens_details=SimpleNamespace(cached_tokens=800),
        completion_tokens_details=SimpleNamespace(reasoning_tokens=10),
    )
    service._client.chat.completions.create.return_value = mock_response

    text, usage = await service.run_inference_with_usage(MagicMock(spec=LLMContext))

    assert text == "hi"
    assert usage.prompt_tokens == 1000
    assert usage.completion_tokens == 50
    assert usage.total_tokens == 1050
    assert usage.cache_read_input_tokens == 800
    assert usage.reasoning_tokens == 10


@pytest.mark.asyncio
async def test_anthropic_run_inference_with_usage_normalizes_cache_tokens():
    """Anthropic input_tokens EXCLUDE cache fields; the seam adds them back."""
    service = AnthropicLLMService(
        api_key="test-key",
        settings=AnthropicLLMService.Settings(model="claude-3-sonnet-20240229"),
    )
    service._client = AsyncMock()
    mock_adapter = MagicMock()
    mock_adapter.get_llm_invocation_params.return_value = AnthropicLLMInvocationParams(
        messages=[{"role": "user", "content": "hi"}], system="sys", tools=[]
    )
    service.get_llm_adapter = MagicMock(return_value=mock_adapter)

    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    mock_response.content[0].text = "hello"
    mock_response.usage = SimpleNamespace(
        input_tokens=100,
        output_tokens=50,
        cache_read_input_tokens=800,
        cache_creation_input_tokens=200,
    )
    service._client.beta.messages.create.return_value = mock_response

    text, usage = await service.run_inference_with_usage(MagicMock(spec=LLMContext))

    assert text == "hello"
    assert usage.prompt_tokens == 1100  # 100 + 800 + 200
    assert usage.completion_tokens == 50
    assert usage.total_tokens == 1150
    assert usage.cache_read_input_tokens == 800
    assert usage.cache_creation_input_tokens == 200


@pytest.mark.asyncio
async def test_google_run_inference_with_usage_includes_thoughts():
    """Gemini thinking tokens bill at the output rate → folded into completion."""
    service = GoogleLLMService(
        api_key="test-key", settings=GoogleLLMService.Settings(model="gemini-2.0-flash")
    )
    service._client = AsyncMock()
    mock_adapter = MagicMock()
    mock_adapter.get_llm_invocation_params.return_value = GeminiLLMInvocationParams(
        messages=[{"role": "user", "content": "hi"}],
        system_instruction="sys",
        tools=NotGiven(),
    )
    service.get_llm_adapter = MagicMock(return_value=mock_adapter)

    mock_response = MagicMock()
    mock_response.candidates = [MagicMock()]
    mock_response.candidates[0].content = MagicMock()
    mock_response.candidates[0].content.parts = [MagicMock()]
    mock_response.candidates[0].content.parts[0].text = "hello"
    mock_response.usage_metadata = SimpleNamespace(
        prompt_token_count=500,
        candidates_token_count=40,
        thoughts_token_count=60,
        total_token_count=600,
        cached_content_token_count=200,
    )
    service._client.aio = AsyncMock()
    service._client.aio.models = AsyncMock()
    service._client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    text, usage = await service.run_inference_with_usage(MagicMock(spec=LLMContext))

    assert text == "hello"
    assert usage.prompt_tokens == 500
    assert usage.completion_tokens == 100  # 40 candidates + 60 thoughts
    assert usage.total_tokens == 600
    assert usage.cache_read_input_tokens == 200
    assert usage.reasoning_tokens == 60


@pytest.mark.asyncio
async def test_aws_bedrock_run_inference_with_usage_normalizes_cache_tokens():
    """Bedrock inputTokens EXCLUDE cache read/write; the seam adds them back."""
    service = AWSBedrockLLMService(
        settings=AWSBedrockLLMService.Settings(
            model="anthropic.claude-3-sonnet-20240229-v1:0"
        )
    )
    mock_adapter = MagicMock()
    mock_adapter.get_llm_invocation_params.return_value = AWSBedrockLLMInvocationParams(
        messages=[{"role": "user", "content": [{"text": "hi"}]}],
        system=[{"text": "sys"}],
        tools=[],
        tool_choice=None,
    )
    service.get_llm_adapter = MagicMock(return_value=mock_adapter)

    mock_client = AsyncMock()
    mock_client.converse.return_value = {
        "output": {"message": {"content": [{"text": "hello"}]}},
        "usage": {
            "inputTokens": 100,
            "outputTokens": 25,
            "cacheReadInputTokens": 300,
            "cacheWriteInputTokens": 50,
        },
    }
    mock_context_manager = AsyncMock()
    mock_context_manager.__aenter__ = AsyncMock(return_value=mock_client)
    mock_context_manager.__aexit__ = AsyncMock(return_value=None)

    with patch.object(service._aws_session, "create_client", return_value=mock_context_manager):
        text, usage = await service.run_inference_with_usage(MagicMock(spec=LLMContext))

    assert text == "hello"
    assert usage.prompt_tokens == 450  # 100 + 300 + 50
    assert usage.completion_tokens == 25
    assert usage.total_tokens == 475
    assert usage.cache_read_input_tokens == 300
    assert usage.cache_creation_input_tokens == 50


@pytest.mark.asyncio
async def test_openai_responses_run_inference_with_usage():
    """Responses-API usage passes through (input already cache-inclusive)."""
    with patch.object(OpenAIResponsesLLMService, "_create_client"):
        service = OpenAIResponsesLLMService(
            settings=OpenAIResponsesLLMService.Settings(
                model="gpt-4.1", system_instruction="sys"
            ),
        )
        service._client = AsyncMock()

        mock_response = MagicMock()
        mock_response.output_text = "hello"
        mock_response.usage = SimpleNamespace(
            input_tokens=700,
            output_tokens=30,
            total_tokens=730,
            input_tokens_details=SimpleNamespace(cached_tokens=600),
            output_tokens_details=SimpleNamespace(reasoning_tokens=5),
        )
        service._client.responses.create = AsyncMock(return_value=mock_response)

        context = LLMContext(messages=[{"role": "user", "content": "hi"}])
        text, usage = await service.run_inference_with_usage(context)

    assert text == "hello"
    assert usage.prompt_tokens == 700
    assert usage.completion_tokens == 30
    assert usage.total_tokens == 730
    assert usage.cache_read_input_tokens == 600
    assert usage.reasoning_tokens == 5


@pytest.mark.asyncio
async def test_run_inference_with_usage_base_delegates_to_run_inference():
    """Subclasses that only override run_inference still serve the usage seam."""
    service = _mocked_openai_service()
    service.run_inference = AsyncMock(return_value="just text")

    text, usage = await LLMService.run_inference_with_usage(
        service, MagicMock(spec=LLMContext)
    )

    assert text == "just text"
    assert usage is None
    service.run_inference.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_inference_delegates_to_run_inference_with_usage():
    """run_inference is a thin delegate: one client call, same text."""
    service = _mocked_openai_service()

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "hi"
    mock_response.usage = None
    service._client.chat.completions.create.return_value = mock_response

    result = await service.run_inference(MagicMock(spec=LLMContext))

    assert result == "hi"
    service._client.chat.completions.create.assert_called_once()
