"""
Integration tests for the LLMService.

These tests verify that the LLMService correctly interacts with
various LLM providers using Pydantic AI.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from airflow_wingman.service.llm_service import LLMService, ToolResult

sys.modules["airflow_wingman.tools"] = MagicMock()
sys.modules["airflow_wingman.tools.pydantic_tools"] = MagicMock()

convert_to_openai_mock = MagicMock(return_value=[{"type": "function", "function": {"name": "test_tool"}}])
convert_to_anthropic_mock = MagicMock(return_value=[{"type": "function", "function": {"name": "test_tool"}}])
convert_to_google_mock = MagicMock(return_value=[{"type": "function", "function": {"name": "test_tool"}}])

sys.modules["airflow_wingman.tools.pydantic_tools"].convert_to_openai_with_pydantic = convert_to_openai_mock
sys.modules["airflow_wingman.tools.pydantic_tools"].convert_to_anthropic_with_pydantic = convert_to_anthropic_mock
sys.modules["airflow_wingman.tools.pydantic_tools"].convert_to_google_with_pydantic = convert_to_google_mock


class WingmanConfig(BaseModel):
    """Configuration for the Wingman service."""

    provider_name: str
    api_key: str
    base_url: str | None = None
    model: str | None = None
    temperature: float = 0.4
    max_tokens: int | None = None


@pytest.fixture
def openai_config():
    """Create a test configuration for OpenAI."""
    return WingmanConfig(provider_name="openai", api_key="test-openai-key", model="gpt-3.5-turbo-test")


@pytest.fixture
def anthropic_config():
    """Create a test configuration for Anthropic."""
    return WingmanConfig(provider_name="anthropic", api_key="test-anthropic-key", model="claude-3-test")


@pytest.fixture
def google_config():
    """Create a test configuration for Google."""
    return WingmanConfig(provider_name="google", api_key="test-google-key", model="gemini-pro-test")


@pytest.fixture
def mock_tool():
    """Create a mock Airflow tool."""
    tool = MagicMock()
    tool.name = "test_tool"
    tool.description = "A test tool"
    tool.inputSchema = {"properties": {"param1": {"type": "string", "description": "Parameter 1"}, "param2": {"type": "integer", "description": "Parameter 2"}}, "required": ["param1"]}
    return tool


class MockOpenAIClient:
    def __init__(self, *args, **kwargs):
        pass

    def complete(self, *args, **kwargs):
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message = MagicMock()
        response.choices[0].message.content = "This is a test response from OpenAI"
        response.choices[0].message.tool_calls = []
        return response

    def complete_stream(self, *args, **kwargs):
        class MockDelta:
            def __init__(self, text):
                self.content = text

        class MockChunk:
            def __init__(self, text):
                self.choices = [MagicMock()]
                self.choices[0].delta = MockDelta(text)

        return [MockChunk("Hello"), MockChunk(" world"), MockChunk("!")]


sys.modules["pydantic_ai"] = MagicMock()
sys.modules["pydantic_ai.openai"] = MagicMock()
sys.modules["pydantic_ai.anthropic"] = MagicMock()
sys.modules["pydantic_ai.google"] = MagicMock()
sys.modules["pydantic_ai.openai"].OpenAIChatCompletion = MockOpenAIClient


class TestLLMService:
    """Test suite for the LLMService."""

    def test_init(self, openai_config):
        """Test initializing the service."""
        service = LLMService(openai_config)
        assert service.provider_name == "openai"
        assert service.api_key == "test-openai-key"
        assert "api_key" in service.provider_config

    def test_setup_provider(self, openai_config, anthropic_config, google_config):
        """Test setting up providers."""
        openai_service = LLMService(openai_config)
        assert "openai" in openai_service.tool_converters

        anthropic_service = LLMService(anthropic_config)
        assert "anthropic" in anthropic_service.tool_converters

        google_service = LLMService(google_config)
        assert "google" in google_service.tool_converters

    def test_set_airflow_tools(self, openai_config, mock_tool):
        """Test setting Airflow tools."""
        service = LLMService(openai_config)
        service.set_airflow_tools([mock_tool])
        assert len(service.airflow_tools) == 1
        assert service.airflow_tools[0].name == "test_tool"

    @patch("airflow_wingman.tools.list_airflow_tools")
    def test_refresh_tools(self, mock_list_tools, openai_config, mock_tool):
        """Test refreshing Airflow tools."""
        mock_list_tools.return_value = [mock_tool]

        service = LLMService(openai_config)
        service.refresh_tools("test-cookie")

        assert len(service.airflow_tools) == 1
        assert service.airflow_tools[0].name == "test_tool"
        mock_list_tools.assert_called_once_with("test-cookie")

    def test_convert_tools(self, openai_config, mock_tool):
        """Test converting tools for providers."""
        service = LLMService(openai_config)
        service.set_airflow_tools([mock_tool])

        with patch("airflow_wingman.tools.pydantic_tools.convert_to_openai_with_pydantic") as mock_convert:
            mock_convert.return_value = [{"type": "function", "function": {"name": "test_tool"}}]
            tools = service._convert_tools()

            assert len(tools) == 1
            assert tools[0]["type"] == "function"
            mock_convert.assert_called_once()

    @patch("pydantic_ai.openai.OpenAIChatCompletion")
    def test_chat_completion_non_streaming(self, mock_openai, openai_config):
        """Test non-streaming chat completion."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = "This is a test response from OpenAI"
        mock_response.choices[0].message.tool_calls = []

        mock_client.complete.return_value = mock_response
        mock_openai.return_value = mock_client

        service = LLMService(openai_config)
        messages = [{"role": "user", "content": "Hello, world!"}]

        response = service.chat_completion(messages=messages, model="gpt-3.5-turbo-test", stream=False)

        assert "content" in response
        assert response["content"] == "This is a test response from OpenAI"
        mock_client.complete.assert_called_once()

    def test_create_tool_result_messages(self, openai_config):
        """Test creating tool result messages."""
        service = LLMService(openai_config)

        results = [ToolResult(tool_call_id="call-1", output="Result 1"), ToolResult(tool_call_id="call-2", output="Result 2")]

        messages = service._create_tool_result_messages(results)

        assert len(messages) == 2
        assert messages[0]["role"] == "tool"
        assert messages[0]["tool_call_id"] == "call-1"
        assert messages[0]["content"] == "Result 1"
        assert messages[1]["role"] == "tool"
        assert messages[1]["tool_call_id"] == "call-2"
        assert messages[1]["content"] == "Result 2"

    def test_from_config(self):
        """Test creating service from config dict."""
        config_dict = {"provider_name": "openai", "api_key": "test-key", "base_url": "https://test-api.com"}

        service = LLMService.from_config(config_dict)

        assert service.provider_name == "openai"
        assert service.api_key == "test-key"
        assert service.base_url == "https://test-api.com"
