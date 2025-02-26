"""
OpenAI provider implementation for Airflow Wingman.

This module contains the OpenAI provider implementation that handles
API requests, tool conversion, and response processing for OpenAI.
"""

import json
import traceback
from typing import Any

from airflow.utils.log.logging_mixin import LoggingMixin
from openai import OpenAI

from airflow_wingman.providers.base import BaseLLMProvider
from airflow_wingman.tools import execute_airflow_tool
from airflow_wingman.tools.conversion import convert_to_openai_tools

# Create a logger instance
logger = LoggingMixin().log


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI provider implementation.

    This class handles API requests, tool conversion, and response processing
    for the OpenAI API. It can also be used for OpenRouter with a custom base URL.
    """

    def __init__(self, api_key: str, base_url: str | None = None):
        """
        Initialize the OpenAI provider.

        Args:
            api_key: API key for OpenAI
            base_url: Optional base URL for the API (used for OpenRouter)
        """
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def convert_tools(self, airflow_tools: list) -> list:
        """
        Convert Airflow tools to OpenAI format.

        Args:
            airflow_tools: List of Airflow tools from MCP server

        Returns:
            List of OpenAI tool definitions
        """
        return convert_to_openai_tools(airflow_tools)

    def create_chat_completion(
        self, messages: list[dict[str, Any]], model: str, temperature: float = 0.7, max_tokens: int | None = None, stream: bool = False, tools: list[dict[str, Any]] | None = None
    ) -> Any:
        """
        Make API request to OpenAI.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model identifier
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            tools: List of tool definitions in OpenAI format

        Returns:
            OpenAI response object

        Raises:
            Exception: If the API request fails
        """
        # Only include tools if we have any
        has_tools = tools is not None and len(tools) > 0
        tool_choice = "auto" if has_tools else None

        try:
            logger.info(f"Sending chat completion request to OpenAI with model: {model}")
            response = self.client.chat.completions.create(
                model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, stream=stream, tools=tools if has_tools else None, tool_choice=tool_choice
            )
            logger.info("Received response from OpenAI")
            return response
        except Exception as e:
            # If the API call fails due to tools not being supported, retry without tools
            error_msg = str(e)
            logger.warning(f"Error in OpenAI API call: {error_msg}")
            if "tools" in error_msg.lower():
                logger.info("Retrying without tools")
                response = self.client.chat.completions.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, stream=stream)
                return response
            else:
                logger.error(f"Failed to get response from OpenAI: {error_msg}\n{traceback.format_exc()}")
                raise

    def has_tool_calls(self, response: Any) -> bool:
        """
        Check if the response contains tool calls.

        Args:
            response: OpenAI response object

        Returns:
            True if the response contains tool calls, False otherwise
        """
        message = response.choices[0].message
        return hasattr(message, "tool_calls") and message.tool_calls

    def process_tool_calls(self, response: Any, cookie: str) -> dict[str, Any]:
        """
        Process tool calls from the response.

        Args:
            response: OpenAI response object
            cookie: Airflow cookie for authentication

        Returns:
            Dictionary mapping tool call IDs to results
        """
        results = {}
        message = response.choices[0].message

        if not self.has_tool_calls(response):
            return results

        for tool_call in message.tool_calls:
            tool_id = tool_call.id
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            try:
                # Execute the Airflow tool with the provided arguments and cookie
                logger.info(f"Executing tool: {function_name} with arguments: {arguments}")
                result = execute_airflow_tool(function_name, arguments, cookie)
                logger.info(f"Tool execution result: {result}")
                results[tool_id] = {"status": "success", "result": result}
            except Exception as e:
                error_msg = f"Error executing tool: {str(e)}\n{traceback.format_exc()}"
                logger.error(error_msg)
                results[tool_id] = {"status": "error", "message": error_msg}

        return results

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
            OpenAI response object
        """
        if not original_response or not tool_results:
            return original_response

        # Get the original message with tool calls
        original_message = original_response.choices[0].message

        # Create a new message with the tool calls
        assistant_message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in original_message.tool_calls],
        }

        # Create tool result messages
        tool_messages = []
        for tool_call_id, result in tool_results.items():
            tool_messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": result.get("result", str(result))})

        # Add the original messages, assistant message, and tool results
        new_messages = messages + [assistant_message] + tool_messages

        # Make a second request to get the final response
        logger.info("Making second request with tool results")
        return self.create_chat_completion(
            messages=new_messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
            tools=None,  # No tools needed for follow-up
        )

    def get_content(self, response: Any) -> str:
        """
        Extract content from the response.

        Args:
            response: OpenAI response object

        Returns:
            Content string from the response
        """
        return response.choices[0].message.content

    def get_streaming_content(self, response: Any) -> Any:
        """
        Get a generator for streaming content from the response.

        Args:
            response: OpenAI streaming response object

        Returns:
            Generator yielding content chunks
        """

        def generate():
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    # Don't do any newline replacement here
                    content = chunk.choices[0].delta.content
                    yield content

        return generate()
