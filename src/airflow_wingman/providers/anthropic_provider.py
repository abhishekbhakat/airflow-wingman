"""
Anthropic provider implementation for Airflow Wingman.

This module contains the Anthropic provider implementation that handles
API requests, tool conversion, and response processing for Anthropic's Claude models.
"""

import json
import logging
import traceback
from typing import Any

from anthropic import Anthropic

from airflow_wingman.providers.base import BaseLLMProvider
from airflow_wingman.tools import execute_airflow_tool
from airflow_wingman.tools.conversion import convert_to_anthropic_tools

# Create a properly namespaced logger for the Airflow plugin
logger = logging.getLogger("airflow.plugins.wingman")


class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic provider implementation.

    This class handles API requests, tool conversion, and response processing
    for the Anthropic API (Claude models).
    """

    def __init__(self, api_key: str):
        """
        Initialize the Anthropic provider.

        Args:
            api_key: API key for Anthropic
        """
        self.api_key = api_key
        self.client = Anthropic(api_key=api_key)

    def convert_tools(self, airflow_tools: list) -> list:
        """
        Convert Airflow tools to Anthropic format.

        Args:
            airflow_tools: List of Airflow tools from MCP server

        Returns:
            List of Anthropic tool definitions
        """
        return convert_to_anthropic_tools(airflow_tools)

    def create_chat_completion(
        self, messages: list[dict[str, Any]], model: str, temperature: float = 0.4, max_tokens: int | None = None, stream: bool = False, tools: list[dict[str, Any]] | None = None
    ) -> Any:
        """
        Make API request to Anthropic.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model identifier
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            tools: List of tool definitions in Anthropic format

        Returns:
            Anthropic response object

        Raises:
            Exception: If the API request fails
        """
        # Convert max_tokens to Anthropic's max_tokens parameter (if provided)
        max_tokens_param = max_tokens if max_tokens is not None else 4096

        # Convert messages from ChatML format to Anthropic's format
        anthropic_messages = self._convert_to_anthropic_messages(messages)

        try:
            logger.info(f"Sending chat completion request to Anthropic with model: {model}")

            # Create request parameters
            params = {"model": model, "messages": anthropic_messages, "temperature": temperature, "max_tokens": max_tokens_param, "stream": stream}

            # Add tools if provided
            if tools and len(tools) > 0:
                params["tools"] = tools
            else:
                logger.warning("No tools included in request")

            # Log the full request parameters (with sensitive information redacted)
            log_params = params.copy()
            logger.info(f"Request parameters: {json.dumps(log_params)}")

            # Make the API request
            response = self.client.messages.create(**params)

            logger.info("Received response from Anthropic")
            return response
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to get response from Anthropic: {error_msg}\n{traceback.format_exc()}")
            raise

    def _convert_to_anthropic_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Convert messages from ChatML format to Anthropic's format.

        Args:
            messages: List of message dictionaries in ChatML format

        Returns:
            List of message dictionaries in Anthropic format
        """
        anthropic_messages = []

        for message in messages:
            role = message["role"]
            content = message["content"]

            # Map ChatML roles to Anthropic roles
            if role == "system":
                # System messages in Anthropic are handled differently
                # We'll add them as a user message with a special prefix
                anthropic_messages.append({"role": "user", "content": f"<system>\n{content}\n</system>"})
            elif role == "user":
                anthropic_messages.append({"role": "user", "content": content})
            elif role == "assistant":
                anthropic_messages.append({"role": "assistant", "content": content})
            elif role == "tool":
                # Tool messages in ChatML become part of the user message in Anthropic
                # We'll handle this in the follow-up completion
                continue

        return anthropic_messages

    def has_tool_calls(self, response: Any) -> bool:
        """
        Check if the response contains tool calls.

        Args:
            response: Anthropic response object

        Returns:
            True if the response contains tool calls, False otherwise
        """
        # Check if any content block is a tool_use block
        if not hasattr(response, "content"):
            return False

        for block in response.content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                return True

        return False

    def process_tool_calls(self, response: Any, cookie: str) -> dict[str, Any]:
        """
        Process tool calls from the response.

        Args:
            response: Anthropic response object
            cookie: Airflow cookie for authentication

        Returns:
            Dictionary mapping tool call IDs to results
        """
        results = {}

        if not self.has_tool_calls(response):
            return results

        # Extract tool_use blocks
        tool_use_blocks = [block for block in response.content if isinstance(block, dict) and block.get("type") == "tool_use"]

        for block in tool_use_blocks:
            tool_id = block.get("id")
            tool_name = block.get("name")
            tool_input = block.get("input", {})

            try:
                # Execute the Airflow tool with the provided arguments and cookie
                logger.info(f"Executing tool: {tool_name} with arguments: {tool_input}")
                result = execute_airflow_tool(tool_name, tool_input, cookie)
                logger.info(f"Tool execution result: {result}")
                results[tool_id] = {"status": "success", "result": result}
            except Exception as e:
                error_msg = f"Error executing tool: {str(e)}\n{traceback.format_exc()}"
                logger.error(error_msg)
                results[tool_id] = {"status": "error", "message": error_msg}

        return results

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
            Anthropic response object
        """
        if not original_response or not tool_results:
            return original_response

        # Extract tool_use blocks from the original response
        tool_use_blocks = [block for block in original_response.content if isinstance(block, dict) and block.get("type") == "tool_use"]

        # Create tool result blocks
        tool_result_blocks = []
        for tool_id, result in tool_results.items():
            tool_result_blocks.append({"type": "tool_result", "tool_use_id": tool_id, "content": result.get("result", str(result))})

        # Convert original messages to Anthropic format
        anthropic_messages = self._convert_to_anthropic_messages(messages)

        # Add the assistant response with tool use
        anthropic_messages.append({"role": "assistant", "content": tool_use_blocks})

        # Add the user message with tool results
        anthropic_messages.append({"role": "user", "content": tool_result_blocks})

        # Make a second request to get the final response
        logger.info("Making second request with tool results")
        return self.create_chat_completion(
            messages=anthropic_messages,
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
            response: Anthropic response object

        Returns:
            Content string from the response
        """
        if not hasattr(response, "content"):
            return ""

        # Combine all text blocks into a single string
        content_parts = []
        for block in response.content:
            if isinstance(block, dict) and block.get("type") == "text":
                content_parts.append(block.get("text", ""))
            elif isinstance(block, str):
                content_parts.append(block)

        return "".join(content_parts)

    def get_streaming_content(self, response: Any) -> Any:
        """
        Get a generator for streaming content from the response.

        Args:
            response: Anthropic streaming response object

        Returns:
            Generator yielding content chunks
        """
        logger.info("Starting Anthropic streaming response processing")

        def generate():
            for chunk in response:
                logger.debug(f"Chunk type: {type(chunk)}")

                # Handle different types of chunks from Anthropic API
                content = None
                if hasattr(chunk, "type") and chunk.type == "content_block_delta":
                    if hasattr(chunk, "delta") and hasattr(chunk.delta, "text"):
                        content = chunk.delta.text
                elif hasattr(chunk, "delta") and hasattr(chunk.delta, "text"):
                    content = chunk.delta.text
                elif hasattr(chunk, "content") and chunk.content:
                    for block in chunk.content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            content = block.get("text", "")

                if content:
                    # Don't do any newline replacement here
                    yield content

        return generate()
