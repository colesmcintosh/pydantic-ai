from __future__ import annotations

import pytest
from pydantic import TypeAdapter

from pydantic_ai.agent import Agent
from pydantic_ai.builtin_tools import (
    AbstractBuiltinTool,
    CodeExecutionTool,
    FileSearchTool,
    UrlContextTool,  # pyright: ignore[reportDeprecated]
    WebFetchTool,
    WebSearchTool,
    XSearchTool,
)
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import Model


@pytest.mark.parametrize('model', ('bedrock', 'mistral', 'cohere', 'huggingface', 'test', 'outlines'), indirect=True)
async def test_builtin_tools_not_supported_web_search(model: Model, allow_model_requests: None):
    agent = Agent(model=model, builtin_tools=[WebSearchTool()])

    with pytest.raises(UserError):
        await agent.run('What day is tomorrow?')


@pytest.mark.parametrize('model', ('bedrock', 'mistral', 'huggingface', 'outlines'), indirect=True)
async def test_builtin_tools_not_supported_web_search_stream(model: Model, allow_model_requests: None):
    agent = Agent(model=model, builtin_tools=[WebSearchTool()])

    with pytest.raises(UserError):
        async with agent.run_stream('What day is tomorrow?'):
            ...  # pragma: no cover


@pytest.mark.parametrize('model', ('groq', 'openai', 'outlines'), indirect=True)
async def test_builtin_tools_not_supported_code_execution(model: Model, allow_model_requests: None):
    agent = Agent(model=model, builtin_tools=[CodeExecutionTool()])

    with pytest.raises(UserError):
        await agent.run('What day is tomorrow?')


@pytest.mark.parametrize('model', ('groq', 'openai', 'outlines'), indirect=True)
async def test_builtin_tools_not_supported_code_execution_stream(model: Model, allow_model_requests: None):
    agent = Agent(model=model, builtin_tools=[CodeExecutionTool()])

    with pytest.raises(UserError):
        async with agent.run_stream('What day is tomorrow?'):
            ...  # pragma: no cover


@pytest.mark.parametrize(
    'model', ('bedrock', 'mistral', 'cohere', 'huggingface', 'groq', 'anthropic', 'test', 'outlines'), indirect=True
)
async def test_builtin_tools_not_supported_file_search(model: Model, allow_model_requests: None):
    agent = Agent(model=model, builtin_tools=[FileSearchTool(file_store_ids=['test-id'])])

    with pytest.raises(UserError):
        await agent.run('Search my files')


@pytest.mark.parametrize('model', ('bedrock', 'mistral', 'huggingface', 'groq', 'anthropic', 'outlines'), indirect=True)
async def test_builtin_tools_not_supported_file_search_stream(model: Model, allow_model_requests: None):
    agent = Agent(model=model, builtin_tools=[FileSearchTool(file_store_ids=['test-id'])])

    with pytest.raises(UserError):
        async with agent.run_stream('Search my files'):
            ...  # pragma: no cover


def test_url_context_tool_is_deprecated():
    """Test that UrlContextTool is deprecated and warns users to use WebFetchTool instead."""
    with pytest.warns(DeprecationWarning, match='Use `WebFetchTool` instead.'):
        UrlContextTool()  # pyright: ignore[reportDeprecated]


def test_url_context_tool_backward_compatibility():
    """Test that old payloads with 'url_context' kind can be deserialized."""
    adapter = TypeAdapter(AbstractBuiltinTool)

    # Test 1: Old payload with url_context should deserialize to UrlContextTool (which is deprecated)
    old_payload = {'kind': 'url_context', 'max_uses': 5, 'enable_citations': True}
    with pytest.warns(DeprecationWarning, match='Use `WebFetchTool` instead.'):
        result = adapter.validate_python(old_payload)
    assert isinstance(result, UrlContextTool)  # pyright: ignore[reportDeprecated]
    assert isinstance(result, WebFetchTool)  # UrlContextTool is a subclass of WebFetchTool
    assert result.kind == 'url_context'  # Preserves the original kind from payload
    assert result.max_uses == 5
    assert result.enable_citations is True

    # Test 2: Re-serialization should preserve the kind
    serialized = adapter.dump_python(result)
    assert serialized['kind'] == 'url_context'
    assert serialized['max_uses'] == 5
    assert serialized['enable_citations'] is True

    # Test 3: New payload with web_fetch should work normally
    new_payload = {'kind': 'web_fetch', 'max_uses': 10}
    result2 = adapter.validate_python(new_payload)
    assert isinstance(result2, WebFetchTool)
    assert result2.kind == 'web_fetch'
    assert result2.max_uses == 10


def test_url_context_tool_instance_behavior():
    """Test that UrlContextTool instances work correctly with deprecation warning."""
    adapter = TypeAdapter(AbstractBuiltinTool)

    # Create instance with deprecation warning
    with pytest.warns(DeprecationWarning, match='Use `WebFetchTool` instead.'):
        tool = UrlContextTool(max_uses=3, enable_citations=True)  # pyright: ignore[reportDeprecated]

    # UrlContextTool inherits from WebFetchTool and overrides kind to 'url_context'
    assert isinstance(tool, WebFetchTool)
    assert tool.kind == 'url_context'
    assert tool.max_uses == 3
    assert tool.enable_citations is True

    # Serialization should use 'url_context'
    serialized = adapter.dump_python(tool)
    assert serialized['kind'] == 'url_context'
    assert serialized['max_uses'] == 3


def test_url_context_discriminated_union():
    """Test that the discriminated union correctly handles both url_context and web_fetch."""
    adapter = TypeAdapter(list[AbstractBuiltinTool])

    # Mix of old and new payloads
    payloads = [
        {'kind': 'url_context', 'max_uses': 1},
        {'kind': 'web_fetch', 'max_uses': 2},
        {'kind': 'web_search'},
        {'kind': 'code_execution'},
    ]

    # Old url_context payloads will trigger deprecation warnings
    with pytest.warns(DeprecationWarning, match='Use `WebFetchTool` instead.'):
        results = adapter.validate_python(payloads)
    assert len(results) == 4
    assert isinstance(results[0], UrlContextTool)  # pyright: ignore[reportDeprecated]
    assert isinstance(results[0], WebFetchTool)  # UrlContextTool is a subclass
    assert results[0].kind == 'url_context'
    assert results[0].max_uses == 1
    assert isinstance(results[1], WebFetchTool)
    assert results[1].kind == 'web_fetch'
    assert results[1].max_uses == 2


# XSearchTool tests


@pytest.mark.parametrize(
    'model',
    ('bedrock', 'mistral', 'cohere', 'huggingface', 'groq', 'anthropic', 'openai', 'test', 'outlines'),
    indirect=True,
)
async def test_builtin_tools_not_supported_x_search(model: Model, allow_model_requests: None):
    """Test that XSearchTool raises UserError for providers that don't support it."""
    agent = Agent(model=model, builtin_tools=[XSearchTool()])

    with pytest.raises(UserError):
        await agent.run('Search X for latest news')


@pytest.mark.parametrize(
    'model', ('bedrock', 'mistral', 'huggingface', 'groq', 'anthropic', 'openai', 'outlines'), indirect=True
)
async def test_builtin_tools_not_supported_x_search_stream(model: Model, allow_model_requests: None):
    """Test that XSearchTool raises UserError for streaming with unsupported providers."""
    agent = Agent(model=model, builtin_tools=[XSearchTool()])

    with pytest.raises(UserError):
        async with agent.run_stream('Search X for latest news'):
            ...  # pragma: no cover


def test_x_search_tool_creation():
    """Test that XSearchTool can be created with various parameters."""
    # Default creation
    tool = XSearchTool()
    assert tool.kind == 'x_search'
    assert tool.allowed_x_handles is None
    assert tool.excluded_x_handles is None
    assert tool.from_date is None
    assert tool.to_date is None
    assert tool.enable_image_understanding is False
    assert tool.enable_video_understanding is False

    # With all parameters
    tool_full = XSearchTool(
        allowed_x_handles=['user1', 'user2'],
        from_date='2025-01-01',
        to_date='2025-06-01',
        enable_image_understanding=True,
        enable_video_understanding=True,
    )
    assert tool_full.allowed_x_handles == ['user1', 'user2']
    assert tool_full.from_date == '2025-01-01'
    assert tool_full.to_date == '2025-06-01'
    assert tool_full.enable_image_understanding is True
    assert tool_full.enable_video_understanding is True


def test_x_search_tool_serialization():
    """Test that XSearchTool can be serialized and deserialized."""
    adapter = TypeAdapter(AbstractBuiltinTool)

    tool = XSearchTool(
        allowed_x_handles=['elonmusk'],
        from_date='2025-01-01',
        enable_image_understanding=True,
    )

    # Serialize
    serialized = adapter.dump_python(tool)
    assert serialized['kind'] == 'x_search'
    assert serialized['allowed_x_handles'] == ['elonmusk']
    assert serialized['from_date'] == '2025-01-01'
    assert serialized['enable_image_understanding'] is True

    # Deserialize
    payload = {
        'kind': 'x_search',
        'excluded_x_handles': ['spammer'],
        'to_date': '2025-12-31',
        'enable_video_understanding': True,
    }
    result = adapter.validate_python(payload)
    assert isinstance(result, XSearchTool)
    assert result.excluded_x_handles == ['spammer']
    assert result.to_date == '2025-12-31'
    assert result.enable_video_understanding is True


def test_x_search_tool_unique_id():
    """Test that XSearchTool has correct unique_id."""
    tool = XSearchTool()
    assert tool.unique_id == 'x_search'


def test_x_search_tool_label():
    """Test that XSearchTool has correct label."""
    tool = XSearchTool()
    assert tool.label == 'X Search'
