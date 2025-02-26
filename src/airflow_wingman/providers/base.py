"""
Base provider interface for Airflow Wingman.

This module contains the base provider interface that all provider implementations
must adhere to. It defines the methods required for tool conversion, API requests,
and response processing.
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseLLMProvider(ABC):
    """
    Base provider interface for LLM providers.

    This abstract class defines the methods that all provider implementations
    must implement to support tool integration.
    """

    @abstractmethod
    def convert_tools(self, airflow_tools: list) -> list:
        """
        Convert internal tool representation to provider format.

        Args:
            airflow_tools: List of Airflow tools from MCP server

        Returns:
            List of provider-specific tool definitions
        """
        pass

    @abstractmethod
    def create_chat_completion(
        self, messages: list[dict[str, Any]], model: str, temperature: float = 0.7, max_tokens: int | None = None, stream: bool = False, tools: list[dict[str, Any]] | None = None
    ) -> Any:
        """
        Make API request to provider.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model identifier
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            tools: List of tool definitions in provider format

        Returns:
            Provider-specific response object
        """
        pass

    @abstractmethod
    def has_tool_calls(self, response: Any) -> bool:
        """
        Check if the response contains tool calls.

        Args:
            response: Provider-specific response object

        Returns:
            True if the response contains tool calls, False otherwise
        """
        pass

    @abstractmethod
    def process_tool_calls(self, response: Any, cookie: str) -> dict[str, Any]:
        """
        Process tool calls from the response.

        Args:
            response: Provider-specific response object
            cookie: Airflow cookie for authentication

        Returns:
            Dictionary mapping tool call IDs to results
        """
        pass

    @abstractmethod
    def create_follow_up_completion(
        self, messages: list[dict[str, Any]], model: str, temperature: float = 0.7, max_tokens: int | None = None, tool_results: dict[str, Any] = None, original_response: Any = None
    ) -> Any:
        """
        Create a follow-up completion with tool results.

        Args:
            messages: Original messages
            model: Model identifier
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            tool_results: Results of tool executions
            original_response: Original response with tool calls

        Returns:
            Provider-specific response object
        """
        pass

    @abstractmethod
    def get_content(self, response: Any) -> str:
        """
        Extract content from the response.

        Args:
            response: Provider-specific response object

        Returns:
            Content string from the response
        """
        pass

    @abstractmethod
    def get_streaming_content(self, response: Any) -> Any:
        """
        Get a generator for streaming content from the response.

        Args:
            response: Provider-specific response object

        Returns:
            Generator yielding content chunks
        """
        pass
