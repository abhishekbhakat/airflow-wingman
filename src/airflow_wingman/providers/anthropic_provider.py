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
            # Log the response (with sensitive information redacted)
            logger.info(f"Anthropic response type: {type(response).__name__}")

            # Log as much information as possible
            if hasattr(response, "json"):
                logger.info(f"Anthropic response json: {json.dumps(response.json)}")

            # Log response attributes
            response_attrs = [attr for attr in dir(response) if not attr.startswith("_") and not callable(getattr(response, attr))]
            logger.info(f"Anthropic response attributes: {response_attrs}")

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
            response: Anthropic response object or generator with tool_call attribute

        Returns:
            True if the response contains tool calls, False otherwise
        """
        # Check if response is a generator with a tool_call attribute
        if hasattr(response, "tool_call") and response.tool_call is not None:
            return True

        # Check if any content block is a tool_use block (for non-streaming responses)
        if hasattr(response, "content"):
            for block in response.content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    return True

        return False

    def process_tool_calls(self, response: Any, cookie: str) -> dict[str, Any]:
        """
        Process tool calls from the response.

        Args:
            response: Anthropic response object or generator with tool_call attribute
            cookie: Airflow cookie for authentication

        Returns:
            Dictionary mapping tool call IDs to results
        """
        results = {}

        if not self.has_tool_calls(response):
            return results

        tool_calls = []

        # Check if response is a generator with a tool_call attribute
        if hasattr(response, "tool_call") and response.tool_call is not None:
            logger.info(f"Processing tool call from generator: {response.tool_call}")
            tool_calls.append(response.tool_call)
        # Otherwise, extract tool calls from response content (for non-streaming responses)
        elif hasattr(response, "content"):
            logger.info("Processing tool calls from response content")
            tool_calls = [block for block in response.content if isinstance(block, dict) and block.get("type") == "tool_use"]

        for tool_call in tool_calls:
            # Extract tool details - handle both formats (generator's tool_call and content block)
            if isinstance(tool_call, dict) and "id" in tool_call:
                # This is from the generator's tool_call attribute
                tool_id = tool_call.get("id")
                tool_name = tool_call.get("name")
                tool_input = tool_call.get("input", {})
            else:
                # This is from the content blocks
                tool_id = tool_call.get("id")
                tool_name = tool_call.get("name")
                tool_input = tool_call.get("input", {})

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
            Generator yielding content chunks with tool_call attribute if detected
        """
        logger.info("Starting Anthropic streaming response processing")

        # Track only the first tool call detected during streaming
        tool_call = None
        tool_use_detected = False

        def generate():
            nonlocal tool_call, tool_use_detected

            for chunk in response:
                logger.debug(f"Chunk type: {type(chunk)}")
                logger.debug(f"Chunk content: {json.dumps(chunk.json) if hasattr(chunk, 'json') else str(chunk)}")

                # Check for content_block_start events with type "tool_use"
                if not tool_use_detected and hasattr(chunk, "type") and chunk.type == "content_block_start":
                    if hasattr(chunk, "content_block") and hasattr(chunk.content_block, "type"):
                        if chunk.content_block.type == "tool_use":
                            logger.info(f"Tool use detected in streaming response: {json.dumps(chunk.json) if hasattr(chunk, 'json') else str(chunk)}")
                            tool_use_detected = True
                            tool_call = {"id": getattr(chunk.content_block, "id", ""), "name": getattr(chunk.content_block, "name", ""), "input": getattr(chunk.content_block, "input", {})}
                            # We don't signal to the frontend during streaming
                            # The tool will only be executed after streaming ends
                            continue

                # Handle content_block_delta events for tool_use (input updates)
                if tool_use_detected and hasattr(chunk, "type") and chunk.type == "content_block_delta":
                    if hasattr(chunk, "delta") and hasattr(chunk.delta, "type") and chunk.delta.type == "input_json_delta":
                        if hasattr(chunk.delta, "partial_json"):
                            logger.info(f"Tool use input update: {chunk.delta.partial_json}")
                            # Update the current tool call input
                            if tool_call:
                                try:
                                    # Try to parse the partial JSON and update the input
                                    partial_input = json.loads(chunk.delta.partial_json)
                                    tool_call["input"].update(partial_input)
                                except json.JSONDecodeError:
                                    logger.warning(f"Failed to parse partial JSON: {chunk.delta.partial_json}")
                            continue

                # Handle content_block_stop events for tool_use
                if tool_use_detected and hasattr(chunk, "type") and chunk.type == "content_block_stop":
                    logger.info("Tool use block completed")
                    # Log the complete tool call for debugging
                    if tool_call:
                        logger.info(f"Completed tool call: {json.dumps(tool_call)}")
                    continue

                # Handle message_delta events with stop_reason "tool_use"
                if hasattr(chunk, "type") and chunk.type == "message_delta":
                    if hasattr(chunk, "delta") and hasattr(chunk.delta, "stop_reason"):
                        if chunk.delta.stop_reason == "tool_use":
                            logger.info("Message stopped due to tool use")
                            continue

                # Handle regular content chunks
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

        # Create the generator
        gen = generate()

        # Attach the single tool_call to the generator object for later reference
        # This will be used after streaming is complete
        gen.tool_call = tool_call

        # Return the enhanced generator
        return gen
