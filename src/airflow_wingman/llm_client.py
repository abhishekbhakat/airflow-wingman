"""
Multi-provider LLM client for Airflow Wingman.

This module contains the LLMClient class that supports multiple LLM providers
(OpenAI, Anthropic, OpenRouter) through a unified interface.
"""

import traceback
from typing import Any

from airflow.utils.log.logging_mixin import LoggingMixin
from flask import session

from airflow_wingman.providers import create_llm_provider
from airflow_wingman.tools import list_airflow_tools

# Create a logger instance
logger = LoggingMixin().log


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

    def chat_completion(self, messages: list[dict[str, str]], model: str, temperature: float = 0.7, max_tokens: int | None = None, stream: bool = False) -> dict[str, Any]:
        """
        Send a chat completion request to the LLM provider.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model identifier
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response (default is True)

        Returns:
            Dictionary with the response content or a generator for streaming
        """
        # Get provider-specific tool definitions from Airflow tools
        provider_tools = self.provider.convert_tools(self.airflow_tools)

        try:
            # Make the initial request with tools
            logger.info(f"Sending chat completion request to {self.provider_name} with model: {model}")
            response = self.provider.create_chat_completion(messages=messages, model=model, temperature=temperature, max_tokens=max_tokens, stream=stream, tools=provider_tools)
            logger.info(f"Received response from {self.provider_name}")

            # If streaming, return the generator directly
            if stream:
                return self.provider.get_streaming_content(response)

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
                    messages=messages, model=model, temperature=temperature, max_tokens=max_tokens, tool_results=tool_results, original_response=response
                )

                return {"content": self.provider.get_content(follow_up_response)}
            else:
                logger.info("Response does not contain tool calls")
                return {"content": self.provider.get_content(response)}

        except Exception as e:
            error_msg = f"Error in {self.provider_name} API call: {str(e)}\\n{traceback.format_exc()}"
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
