"""
LLM Service using Pydantic AI for Airflow Wingman.

This module provides a unified service for interacting with various LLM providers
through Pydantic AI, supporting both synchronous and asynchronous operations.
"""

import logging
import traceback
from collections.abc import AsyncGenerator, Generator
from typing import Any

from airflow_wingman.models.config import WingmanConfig
from airflow_wingman.models.tools import ToolResult
from airflow_wingman.tools import execute_tool, list_airflow_tools

logger = logging.getLogger("airflow.plugins.wingman")


class LLMService:
    """
    LLM service using Pydantic AI.

    This service provides a unified interface for interacting with different
    LLM providers (OpenAI, Anthropic, Google) using Pydantic AI.
    """

    def __init__(self, config: WingmanConfig):
        """
        Initialize the LLM service.

        Args:
            config: Wingman configuration containing provider details
        """
        self.config = config
        self.provider_name = config.provider_name
        self.api_key = config.api_key
        self.base_url = config.base_url
        self.airflow_tools = []

        self._setup_provider()

        logger.info(f"Initialized LLM service with provider: {self.provider_name}")

    def _setup_provider(self):
        """Set up provider-specific configurations."""
        self.provider_config = {
            "api_key": self.api_key,
        }

        if self.base_url:
            self.provider_config["base_url"] = self.base_url

    def set_airflow_tools(self, tools: list[Any]):
        """
        Set the available Airflow tools.

        Args:
            tools: List of Airflow Tool objects
        """
        self.airflow_tools = tools
        logger.info(f"Set {len(tools)} Airflow tools")

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

    def _prepare_chat_request(self, messages: list[dict[str, Any]], model: str, temperature: float, max_tokens: int | None, stream: bool, tools: list[Any] | None = None):
        """
        Prepare a chat request for the appropriate provider using Pydantic AI.

        Args:
            messages: List of message dictionaries
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            tools: Optional list of tools to include

        Returns:
            Provider-specific chat request
        """
        request_config = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }

        if max_tokens:
            request_config["max_tokens"] = max_tokens

        if tools and len(tools) > 0:
            request_config["tools"] = tools
        elif self.airflow_tools:
            # Pydantic AI will handle the conversion automatically
            request_config["tools"] = self.airflow_tools

        return request_config

    def _get_client(self, async_mode: bool = False):
        """
        Get the appropriate Pydantic AI client based on the provider.

        Args:
            async_mode: Whether to return an async client

        Returns:
            Pydantic AI client instance
        """
        if self.provider_name.lower() == "openai":
            if async_mode:
                from pydantic_ai.openai import AsyncOpenAIChatCompletion

                return AsyncOpenAIChatCompletion(**self.provider_config)
            else:
                from pydantic_ai.openai import OpenAIChatCompletion

                return OpenAIChatCompletion(**self.provider_config)

        elif self.provider_name.lower() == "anthropic":
            if async_mode:
                from pydantic_ai.anthropic import AsyncAnthropicChatCompletion

                return AsyncAnthropicChatCompletion(**self.provider_config)
            else:
                from pydantic_ai.anthropic import AnthropicChatCompletion

                return AnthropicChatCompletion(**self.provider_config)

        elif self.provider_name.lower() == "google":
            if async_mode:
                from pydantic_ai.google import AsyncGoogleChatCompletion

                return AsyncGoogleChatCompletion(**self.provider_config)
            else:
                from pydantic_ai.google import GoogleChatCompletion

                return GoogleChatCompletion(**self.provider_config)

        else:
            raise ValueError(f"Unsupported provider: {self.provider_name}")

    def chat_completion(self, messages: list[dict[str, Any]], model: str, temperature: float = 0.4, max_tokens: int | None = None, stream: bool = True) -> dict[str, Any] | Generator[str, None, None]:
        """
        Send a chat completion request to the LLM provider.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model identifier
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response

        Returns:
            If stream=False: Dictionary with the response content
            If stream=True: Generator for streaming
        """
        try:
            client = self._get_client()

            request_config = self._prepare_chat_request(messages=messages, model=model, temperature=temperature, max_tokens=max_tokens, stream=stream)

            logger.info(f"Sending chat completion request to {self.provider_name} with model: {model}")

            if stream:
                stream_generator = client.complete_stream(**request_config)

                def content_generator():
                    try:
                        for chunk in stream_generator:
                            if hasattr(chunk, "choices") and chunk.choices and hasattr(chunk.choices[0], "delta"):
                                delta = chunk.choices[0].delta
                                if hasattr(delta, "content") and delta.content:
                                    yield delta.content
                    except Exception as e:
                        logger.error(f"Error streaming response: {str(e)}")
                        yield f"Error: {str(e)}"

                return content_generator()
            else:
                response = client.complete(**request_config)
                logger.info(f"Received response from {self.provider_name}")

                if hasattr(response, "choices") and response.choices:
                    choice = response.choices[0]
                    if hasattr(choice, "message") and hasattr(choice.message, "content"):
                        return {"content": choice.message.content}

                return {"content": str(response)}

        except Exception as e:
            error_msg = f"Error in chat completion: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)

            if stream:
                error_message = str(e)  # Capture the exception message

                def error_generator():
                    yield f"Error: {error_message}"

                return error_generator()
            else:
                return {"error": str(e)}

    async def async_chat_completion(
        self, messages: list[dict[str, Any]], model: str, temperature: float = 0.4, max_tokens: int | None = None, stream: bool = True
    ) -> dict[str, Any] | AsyncGenerator[str, None]:
        """
        Send an asynchronous chat completion request to the LLM provider.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model identifier
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response

        Returns:
            If stream=False: Dictionary with the response content
            If stream=True: AsyncGenerator for streaming
        """
        try:
            client = self._get_client(async_mode=True)

            request_config = self._prepare_chat_request(messages=messages, model=model, temperature=temperature, max_tokens=max_tokens, stream=stream)

            logger.info(f"Sending async chat completion request to {self.provider_name} with model: {model}")

            if stream:
                stream_generator = client.acomplete_stream(**request_config)

                async def content_generator():
                    try:
                        async for chunk in stream_generator:
                            if hasattr(chunk, "choices") and chunk.choices and hasattr(chunk.choices[0], "delta"):
                                delta = chunk.choices[0].delta
                                if hasattr(delta, "content") and delta.content:
                                    yield delta.content
                    except Exception as e:
                        logger.error(f"Error in async streaming: {str(e)}")
                        yield f"Error: {str(e)}"

                return content_generator()
            else:
                response = await client.acomplete(**request_config)
                logger.info(f"Received async response from {self.provider_name}")

                if hasattr(response, "choices") and response.choices:
                    choice = response.choices[0]
                    if hasattr(choice, "message") and hasattr(choice.message, "content"):
                        return {"content": choice.message.content}

                return {"content": str(response)}

        except Exception as e:
            error_msg = f"Error in async chat completion: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)

            if stream:
                error_message = str(e)  # Capture the exception message

                async def error_generator():
                    yield f"Error: {error_message}"

                return error_generator()
            else:
                return {"error": str(e)}

    def process_tool_calls_and_follow_up(
        self,
        response: Any,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float = 0.4,
        max_tokens: int | None = None,
        cookie: str | None = None,
        max_iterations: int = 5,
        stream: bool = True,
    ) -> Generator[str, None, None]:
        """
        Process tool calls recursively and make follow-up requests.

        Args:
            response: Previous response containing tool calls (can be None if starting fresh)
            messages: List of message dictionaries
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            cookie: Airflow cookie for authentication
            max_iterations: Maximum number of tool call iterations
            stream: Whether to stream the final response

        Returns:
            Generator yielding the final response
        """
        if not cookie:
            error_msg = "No Airflow cookie available for tool execution"
            logger.error(error_msg)
            yield f"Error: {error_msg}"
            return

        try:
            client = self._get_client()

            request_config = self._prepare_chat_request(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,  # Start with non-streaming to process tool calls
            )

            logger.info(f"Starting tool call processing with {self.provider_name}")

            # If no previous response is provided, get a new one
            current_response = response if response else client.complete(**request_config)

            iteration = 0

            while iteration < max_iterations:
                iteration += 1
                logger.info(f"Tool call iteration {iteration}/{max_iterations}")

                if not self._has_tool_calls(current_response):
                    logger.info("No tool calls in response, returning content")
                    content = self._get_response_content(current_response)
                    yield content
                    return

                tool_results = self._process_tool_calls(current_response, cookie)

                tool_messages = self._create_tool_result_messages(tool_results)
                updated_messages = messages + tool_messages

                follow_up_config = self._prepare_chat_request(
                    messages=updated_messages, model=model, temperature=temperature, max_tokens=max_tokens, stream=stream if iteration == max_iterations else False
                )

                if iteration == max_iterations or stream:
                    stream_generator = client.complete_stream(**follow_up_config)
                    for chunk in stream_generator:
                        if hasattr(chunk, "choices") and chunk.choices and hasattr(chunk.choices[0], "delta"):
                            delta = chunk.choices[0].delta
                            if hasattr(delta, "content") and delta.content:
                                yield delta.content
                    return
                else:
                    current_response = client.complete(**follow_up_config)

            logger.warning(f"Reached maximum tool call iterations ({max_iterations})")
            content = self._get_response_content(current_response)
            yield content

        except Exception as e:
            error_msg = f"Error processing tool calls: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            yield f"Error: {str(e)}"

    def _has_tool_calls(self, response: Any) -> bool:
        """
        Check if a response contains tool calls.

        Args:
            response: Response from Pydantic AI

        Returns:
            True if the response contains tool calls, False otherwise
        """
        if not hasattr(response, "choices") or not response.choices:
            return False

        choice = response.choices[0]
        if not hasattr(choice, "message"):
            return False

        return hasattr(choice.message, "tool_calls") and bool(choice.message.tool_calls)

    def _get_response_content(self, response: Any) -> str:
        """
        Extract content from a response.

        Args:
            response: Response from Pydantic AI

        Returns:
            Content string
        """
        try:
            if hasattr(response, "choices") and response.choices:
                choice = response.choices[0]
                if hasattr(choice, "message") and hasattr(choice.message, "content"):
                    return choice.message.content or ""
        except Exception as e:
            logger.error(f"Error extracting content: {str(e)}")

        return ""

    def _process_tool_calls(self, response: Any, cookie: str) -> list[ToolResult]:
        """
        Process tool calls from a response.

        Args:
            response: Response containing tool calls
            cookie: Airflow cookie for authentication

        Returns:
            List of tool results
        """
        if not self._has_tool_calls(response):
            return []

        tool_calls = response.choices[0].message.tool_calls
        logger.info(f"Processing {len(tool_calls)} tool calls")

        results = []
        for tool_call in tool_calls:
            try:
                function_name = tool_call.function.name
                function_args = tool_call.function.arguments

                result = execute_tool(name=function_name, arguments=function_args, cookie=cookie)

                tool_result = ToolResult(tool_name=function_name, tool_args=function_args, result=result, status="success")

                results.append(tool_result)
                logger.info(f"Successfully executed tool: {function_name}")

            except Exception as e:
                error_msg = f"Error executing tool: {str(e)}"
                logger.error(error_msg)

                tool_result = ToolResult(
                    tool_name=function_name if hasattr(tool_call.function, "name") else "unknown",
                    tool_args=function_args if hasattr(tool_call.function, "arguments") else {},
                    result=None,
                    error=str(e),
                    status="error",
                )

                results.append(tool_result)

        return results

    def _create_tool_result_messages(self, tool_results: list[ToolResult]) -> list[dict[str, Any]]:
        """
        Create messages for tool results.

        Args:
            tool_results: List of tool execution results

        Returns:
            List of message dictionaries for the tool results
        """
        messages = []

        for result in tool_results:
            # Format the message according to the LLM provider expectations
            content = result.result if result.is_success else f"Error: {result.error}"
            message = {"role": "tool", "content": str(content), "name": result.tool_name}

            messages.append(message)

        return messages

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "LLMService":
        """
        Create an LLMService instance from a configuration dictionary.

        Args:
            config: Configuration dictionary with provider_name, api_key,
                   and optional base_url

        Returns:
            LLMService instance
        """
        wingman_config = WingmanConfig(
            provider_name=config.get("provider_name", "openai"),
            api_key=config.get("api_key", ""),
            base_url=config.get("base_url"),
            model=config.get("model", "gpt-3.5-turbo-0125"),
            temperature=config.get("temperature", 0.4),
            max_tokens=config.get("max_tokens"),
        )

        return cls(wingman_config)
