"""
Multi-provider LLM client for Airflow Wingman.

This module contains the LLMClient class that supports multiple LLM providers
(OpenAI, Anthropic, OpenRouter) through a unified interface.
"""

import logging
import traceback
from typing import Any

from flask import session

from airflow_wingman.providers import create_llm_provider
from airflow_wingman.tools import list_airflow_tools

# Create a properly namespaced logger for the Airflow plugin
logger = logging.getLogger("airflow.plugins.wingman")


class LLMClient:
    """
    Multi-provider LLM client for Airflow Wingman.

    This class handles chat completion requests to various LLM providers
    (OpenAI, Anthropic, OpenRouter) through a unified interface.
    """

    def __init__(self, provider_name: str, api_key: str, base_url: str | None = None):
        """
        Initialize the LLM client.

        Args:
            provider_name: Name of the provider (openai, anthropic, openrouter)
            api_key: API key for the provider
            base_url: Optional base URL for the provider API
        """
        self.provider_name = provider_name
        self.api_key = api_key
        self.base_url = base_url
        self.provider = create_llm_provider(provider_name, api_key, base_url)
        self.airflow_tools = []

    def set_airflow_tools(self, tools: list):
        """
        Set the available Airflow tools.

        Args:
            tools: List of Airflow Tool objects
        """
        self.airflow_tools = tools

    def chat_completion(
        self, messages: list[dict[str, str]], model: str, temperature: float = 0.4, max_tokens: int | None = None, stream: bool = True, return_response_obj: bool = False
    ) -> dict[str, Any] | tuple[Any, Any]:
        """
        Send a chat completion request to the LLM provider.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model identifier
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response (default is True)
            return_response_obj: If True and streaming, returns both the response object and generator

        Returns:
            If stream=False: Dictionary with the response content
            If stream=True and return_response_obj=False: Generator for streaming
            If stream=True and return_response_obj=True: Tuple of (response_obj, generator)
        """
        # Get provider-specific tool definitions from Airflow tools
        provider_tools = self.provider.convert_tools(self.airflow_tools)

        try:
            # Make the initial request with tools
            logger.info(f"Sending chat completion request to {self.provider_name} with model: {model}")
            response = self.provider.create_chat_completion(messages=messages, model=model, temperature=temperature, max_tokens=max_tokens, stream=stream, tools=provider_tools)
            logger.info(f"Received response from {self.provider_name}")

            # If streaming, handle based on return_response_obj flag
            if stream:
                logger.info(f"Using streaming response from {self.provider_name}")
                streaming_content = self.provider.get_streaming_content(response)
                return streaming_content

            # For non-streaming responses, handle tool calls if present
            if self.provider.has_tool_calls(response):
                logger.info("Response contains tool calls")

                # Process tool calls and get results
                cookie = session.get("airflow_cookie")
                if not cookie:
                    error_msg = "No Airflow cookie available"
                    logger.error(error_msg)
                    return {"error": error_msg}

                tool_results = self.provider.process_tool_calls(response, cookie)

                # Create a follow-up completion with the tool results
                logger.info("Making follow-up request with tool results")
                follow_up_response = self.provider.create_follow_up_completion(
                    messages=messages, model=model, temperature=temperature, max_tokens=max_tokens, tool_results=tool_results, original_response=response, tools=provider_tools
                )

                content = self.provider.get_content(follow_up_response)
                logger.info(f"Final content from {self.provider_name} with tool calls COMPLETE RESPONSE START >>>")
                logger.info(content)
                logger.info("<<< COMPLETE RESPONSE END")
                return {"content": content}
            else:
                logger.info("Response does not contain tool calls")
                content = self.provider.get_content(response)
                logger.info(f"Final content from {self.provider_name} without tool calls COMPLETE RESPONSE START >>>")
                logger.info(content)
                logger.info("<<< COMPLETE RESPONSE END")
                return {"content": content}

        except Exception as e:
            error_msg = f"Error in {self.provider_name} API call: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            return {"error": f"API request failed: {str(e)}"}

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "LLMClient":
        """
        Create an LLMClient instance from a configuration dictionary.

        Args:
            config: Configuration dictionary with provider_name, api_key, and optional base_url

        Returns:
            LLMClient instance
        """
        provider_name = config.get("provider_name", "openai")
        api_key = config.get("api_key")
        base_url = config.get("base_url")

        if not api_key:
            raise ValueError("API key is required")

        return cls(provider_name=provider_name, api_key=api_key, base_url=base_url)

    def process_tool_calls_and_follow_up(self, response, messages, model, temperature, max_tokens, max_iterations=5, cookie=None, stream=True):
        """
        Process tool calls recursively from a response and make follow-up requests until
        there are no more tool calls or max_iterations is reached.
        Returns a generator for streaming the final follow-up response.

        Args:
            response: The original response object containing tool calls
            messages: List of message dictionaries with 'role' and 'content'
            model: Model identifier
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            max_iterations: Maximum number of tool call iterations to prevent infinite loops
            cookie: Airflow cookie for authentication (optional, will try to get from session if not provided)
            stream: Whether to stream the response

        Returns:
            Generator for streaming the final follow-up response
        """
        try:
            iteration = 0
            current_response = response

            # Check if we have a cookie
            if not cookie:
                error_msg = "No Airflow cookie available"
                logger.error(error_msg)
                yield f"Error: {error_msg}"
                return

            # Process tool calls recursively until there are no more or max_iterations is reached
            while self.provider.has_tool_calls(current_response) and iteration < max_iterations:
                iteration += 1
                logger.info(f"Processing tool calls iteration {iteration}/{max_iterations}")

                # Process tool calls and get results
                tool_results = self.provider.process_tool_calls(current_response, cookie)

                # Make follow-up request with tool results
                logger.info(f"Making follow-up request with tool results (iteration {iteration})")

                # Always stream follow-up requests to ensure consistent behavior
                # This ensures we get streaming responses from the provider
                should_stream = True
                logger.info(f"Setting should_stream=True for follow-up request (iteration {iteration})")

                # Get provider-specific tool definitions from Airflow tools
                provider_tools = self.provider.convert_tools(self.airflow_tools)

                follow_up_response = self.provider.create_follow_up_completion(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tool_results=tool_results,
                    original_response=current_response,
                    stream=should_stream,
                    tools=provider_tools,
                )

                # Check if this follow-up response has more tool calls
                if not self.provider.has_tool_calls(follow_up_response):
                    logger.info(f"No more tool calls after iteration {iteration}")
                    # Final response - always yield content in a streaming fashion
                    # Since we're always streaming now, we can directly yield chunks from the streaming generator
                    chunk_count = 0
                    for chunk in self.provider.get_streaming_content(follow_up_response):
                        chunk_count += 1
                        # logger.info(f"Yielding chunk {chunk_count} from streaming generator: {chunk[:50] if chunk else 'Empty chunk'}...")
                        yield chunk
                    logger.info(f"Finished yielding {chunk_count} chunks from streaming generator")

                # Update current_response for the next iteration
                current_response = follow_up_response

            # If we've reached max_iterations and still have tool calls, log a warning
            if iteration == max_iterations and self.provider.has_tool_calls(current_response):
                logger.warning(f"Reached maximum tool call iterations ({max_iterations})")
                # Stream the final response even if it has tool calls
                if not should_stream:
                    # If we didn't stream this response, convert it to a single chunk
                    content = self.provider.get_content(follow_up_response)
                    logger.info(f"Yielding complete content as a single chunk (max iterations): {content[:100]}...")
                    yield content
                    logger.info("Finished yielding complete content (max iterations)")
                else:
                    # Yield chunks from the streaming generator
                    logger.info("Starting to yield chunks from streaming generator (max iterations reached)")
                    chunk_count = 0
                    for chunk in self.provider.get_streaming_content(follow_up_response):
                        chunk_count += 1
                        logger.info(f"Yielding chunk {chunk_count} from streaming generator (max iterations)")
                        yield chunk
                    logger.info(f"Finished yielding {chunk_count} chunks from streaming generator (max iterations)")

            # If we didn't process any tool calls (shouldn't happen), return an error
            if iteration == 0:
                error_msg = "No tool calls found in response"
                logger.error(error_msg)
                yield f"Error: {error_msg}"

        except Exception as e:
            error_msg = f"Error processing tool calls: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            yield f"Error: {str(e)}"

    def refresh_tools(self, cookie: str) -> None:
        """
        Refresh the available Airflow tools.

        Args:
            cookie: Airflow cookie for authentication
        """
        try:
            logger.info("Refreshing Airflow tools")
            tools = list_airflow_tools(cookie)
            self.set_airflow_tools(tools)
            logger.info(f"Refreshed {len(tools)} Airflow tools")
        except Exception as e:
            error_msg = f"Error refreshing Airflow tools: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            # Don't raise the exception, just log it
            # The client will continue to use the existing tools (if any)
