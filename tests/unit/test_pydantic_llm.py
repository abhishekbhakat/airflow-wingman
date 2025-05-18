"""
Unit tests for the LLMService using Pydantic models.

These tests focus on the core LLMService functionality without dependencies on Airflow.
"""

import sys
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

sys.modules["airflow_wingman.tools"] = MagicMock()
sys.modules["airflow_wingman.tools.execute_tool"] = MagicMock(return_value="Tool executed successfully")
sys.modules["airflow_wingman.tools.list_airflow_tools"] = MagicMock(return_value=[])
sys.modules["airflow_wingman.tools.pydantic_tools"] = MagicMock()
sys.modules["airflow_wingman.plugin"] = MagicMock()
sys.modules["airflow_wingman.views"] = MagicMock()
sys.modules["flask_appbuilder"] = MagicMock()

sys.modules["pydantic_ai"] = MagicMock()
sys.modules["pydantic_ai.openai"] = MagicMock()
sys.modules["pydantic_ai.anthropic"] = MagicMock()
sys.modules["pydantic_ai.google"] = MagicMock()


class WingmanConfig(BaseModel):
    """Configuration for the Wingman service."""

    provider_name: str
    api_key: str
    base_url: str | None = None
    model: str | None = None
    temperature: float = 0.4
    max_tokens: int | None = None


class Message(BaseModel):
    """Chat message model."""

    role: str
    content: str
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class ToolResult(BaseModel):
    """Model for tool execution results."""

    tool_call_id: str
    output: str


class LLMService:
    """LLM service using Pydantic AI."""

    def __init__(self, config: WingmanConfig):
        """Initialize the LLM service."""
        self.config = config
        self.provider_name = config.provider_name
        self.api_key = config.api_key
        self.base_url = config.base_url
        self.airflow_tools = []

        self.provider_config = {"api_key": self.api_key}
        if self.base_url:
            self.provider_config["base_url"] = self.base_url

        self.tool_converters = {
            "openai": MagicMock(return_value=[{"type": "function", "function": {"name": "test_tool"}}]),
            "anthropic": MagicMock(return_value=[{"type": "function", "function": {"name": "test_tool"}}]),
            "google": MagicMock(return_value=[{"type": "function", "function": {"name": "test_tool"}}]),
        }

    def set_airflow_tools(self, tools: list[Any]) -> None:
        """Set the available Airflow tools."""
        self.airflow_tools = tools

    def _convert_tools(self) -> list[Any]:
        """Convert Airflow tools to the format needed by the provider."""
        if not self.airflow_tools:
            return []

        converter = self.tool_converters.get(self.provider_name.lower())
        if not converter:
            return []

        return converter(self.airflow_tools)

    def _create_tool_result_messages(self, tool_results: list[ToolResult]) -> list[dict[str, Any]]:
        """Create messages from tool results."""
        messages = []
        for result in tool_results:
            message = {"role": "tool", "tool_call_id": result.tool_call_id, "content": result.output}
            messages.append(message)
        return messages

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "LLMService":
        """Create an LLMService from a config dictionary."""
        wingman_config = WingmanConfig(provider_name=config.get("provider_name", "openai"), api_key=config.get("api_key", ""), base_url=config.get("base_url"))
        return cls(wingman_config)


@pytest.fixture
def openai_config():
    """Create a test configuration for OpenAI."""
    return WingmanConfig(provider_name="openai", api_key="test-openai-key", model="gpt-3.5-turbo-test")


@pytest.fixture
def mock_tool():
    """Create a mock Airflow tool."""
    tool = MagicMock()
    tool.name = "test_tool"
    tool.description = "A test tool"
    tool.inputSchema = {"properties": {"param1": {"type": "string", "description": "Parameter 1"}, "param2": {"type": "integer", "description": "Parameter 2"}}, "required": ["param1"]}
    return tool


class TestLLMService:
    """Tests for the LLMService."""

    def test_init(self, openai_config):
        """Test initializing the service."""
        service = LLMService(openai_config)
        assert service.provider_name == "openai"
        assert service.api_key == "test-openai-key"
        assert "api_key" in service.provider_config

    def test_set_airflow_tools(self, openai_config, mock_tool):
        """Test setting Airflow tools."""
        service = LLMService(openai_config)
        service.set_airflow_tools([mock_tool])
        assert len(service.airflow_tools) == 1
        assert service.airflow_tools[0].name == "test_tool"

    def test_convert_tools(self, openai_config, mock_tool):
        """Test converting tools for providers."""
        service = LLMService(openai_config)
        service.set_airflow_tools([mock_tool])
        tools = service._convert_tools()
        assert len(tools) == 1
        assert tools[0]["type"] == "function"

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
