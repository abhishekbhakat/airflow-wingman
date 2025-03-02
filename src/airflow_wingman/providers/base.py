"""
Base provider interface for Airflow Wingman.

This module contains the base provider interface that all provider implementations
must adhere to. It defines the methods required for tool conversion, API requests,
and response processing.
"""

import json
from abc import ABC, abstractmethod
from collections.abc import Generator, Iterator
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class StreamingResponse(Generic[T]):
    """
    Wrapper for streaming responses that can hold tool call information.

    This class wraps a generator and provides an iterator interface while also
    storing tool call information. This allows us to associate metadata with
    a generator without modifying the generator itself.
    """

    def __init__(self, generator: Generator[T, None, None], tool_call: dict = None):
        """
        Initialize the streaming response.

        Args:
            generator: The underlying generator yielding content chunks
            tool_call: Optional tool call information detected during streaming
        """
        self.generator = generator
        self.tool_call = tool_call

    def __iter__(self) -> Iterator[T]:
        """
        Return self as iterator.
        """
        return self

    def __next__(self) -> T:
        """
        Get the next item from the generator.
        """
        return next(self.generator)


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
        self, messages: list[dict[str, Any]], model: str, temperature: float = 0.4, max_tokens: int | None = None, stream: bool = False, tools: list[dict[str, Any]] | None = None
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

    def has_tool_calls(self, response: Any) -> bool:
        """
        Check if the response contains tool calls.

        Args:
            response: Provider-specific response object or StreamingResponse

        Returns:
            True if the response contains tool calls, False otherwise
        """
        # Check if response is a StreamingResponse with a tool_call attribute
        if isinstance(response, StreamingResponse) and response.tool_call is not None:
            return True

        # Provider-specific implementation should handle other cases
        return False

    def get_tool_calls(self, response: Any) -> list:
        """
        Extract tool calls from the response.

        Args:
            response: Provider-specific response object or StreamingResponse

        Returns:
            List of tool call objects in a standardized format
        """
        tool_calls = []

        # Check if response is a StreamingResponse with a tool_call attribute
        if isinstance(response, StreamingResponse) and response.tool_call is not None:
            tool_calls.append(response.tool_call)

        # Provider-specific implementation should handle other cases
        return tool_calls

    def process_tool_calls(self, response: Any, cookie: str) -> dict[str, Any]:
        """
        Process tool calls from the response.

        Args:
            response: Provider-specific response object or StreamingResponse
            cookie: Airflow cookie for authentication

        Returns:
            Dictionary mapping tool call IDs to results
        """
        tool_calls = self.get_tool_calls(response)
        results = {}

        for tool_call in tool_calls:
            tool_name = tool_call.get("name", "")
            tool_input = tool_call.get("input", {})
            tool_id = tool_call.get("id", "")

            try:
                import logging

                logger = logging.getLogger(__name__)
                logger.info(f"Executing tool: {tool_name} with input: {json.dumps(tool_input)}")

                from airflow_wingman.tools import execute_airflow_tool

                result = execute_airflow_tool(tool_name, tool_input, cookie)

                logger.info(f"Tool result: {json.dumps(result)}")
                results[tool_id] = {
                    "name": tool_name,
                    "input": tool_input,
                    "output": result,
                }
            except Exception as e:
                import traceback

                error_msg = f"Error executing tool: {str(e)}\n{traceback.format_exc()}"
                logger.error(error_msg)
                results[tool_id] = {"status": "error", "message": error_msg}

        return results

    @abstractmethod
    def create_follow_up_completion(
        self, messages: list[dict[str, Any]], model: str, temperature: float = 0.4, max_tokens: int | None = None, tool_results: dict[str, Any] = None, original_response: Any = None
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
    def get_streaming_content(self, response: Any) -> StreamingResponse:
        """
        Get a StreamingResponse for streaming content from the response.

        Args:
            response: Provider-specific response object

        Returns:
            StreamingResponse object wrapping a generator that yields content chunks
            and can also store tool call information detected during streaming
        """
        pass
