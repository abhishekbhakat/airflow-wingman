"""
Client interface for Airflow Wingman using Pydantic AI.

This module provides a high-level client interface that uses the new LLMService
under the hood, with a simplified API that maintains backward compatibility.
"""

import logging
from collections.abc import Generator
from typing import Any

from airflow_wingman.config import WingmanConfig
from airflow_wingman.service.llm_service import LLMService

logger = logging.getLogger("airflow.plugins.wingman")


class WingmanClient:
    """
    High-level client for Airflow Wingman using Pydantic AI.

    This client provides a simplified interface to the LLM service while
    maintaining backward compatibility with the previous LLMClient.
    """

    def __init__(
        self,
        provider_name: str,
        api_key: str,
        base_url: str | None = None,
        model: str | None = None,
    ):
        """
        Initialize the Wingman client.

        Args:
            provider_name: Name of the provider (openai, anthropic, google)
            api_key: API key for the provider
            base_url: Optional base URL for the provider API
            model: Optional default model to use
        """
        self.config = WingmanConfig(
            provider_name=provider_name,
            api_key=api_key,
            base_url=base_url,
        )

        if model:
            self.config.model = model

        self.service = LLMService(self.config)
        self.airflow_tools = []

        logger.info(f"Initialized Wingman client with provider: {provider_name}")

    def set_airflow_tools(self, tools: list[Any]) -> None:
        """
        Set the available Airflow tools.

        Args:
            tools: List of Airflow Tool objects
        """
        self.service.set_airflow_tools(tools)
        self.airflow_tools = tools

    def refresh_tools(self, cookie: str) -> None:
        """
        Refresh the available Airflow tools.

        Args:
            cookie: Airflow cookie for authentication
        """
        self.service.refresh_tools(cookie)
        self.airflow_tools = self.service.airflow_tools

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.4,
        max_tokens: int | None = None,
        stream: bool = True,
        return_response_obj: bool = False,
    ) -> dict[str, Any] | Generator[str, None, None] | tuple:
        """
        Send a chat completion request to the LLM provider.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model identifier (overrides the default model)
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            return_response_obj: If True and streaming, returns a tuple with
                                 (None, generator) for backward compatibility

        Returns:
            If stream=False: Dictionary with the response content
            If stream=True and return_response_obj=False: Generator for streaming
            If stream=True and return_response_obj=True: Tuple of (None, generator)
        """
        model_to_use = model or self.config.model

        try:
            logger.info(f"Sending chat completion request with model: {model_to_use}")

            if stream:
                stream_generator = self.service.chat_completion(messages=messages, model=model_to_use, temperature=temperature, max_tokens=max_tokens, stream=True)

                if return_response_obj:
                    return None, stream_generator
                else:
                    return stream_generator
            else:
                response = self.service.chat_completion(messages=messages, model=model_to_use, temperature=temperature, max_tokens=max_tokens, stream=False)

                logger.info("Received non-streaming response")
                return response

        except Exception as e:
            logger.error(f"Error in chat completion: {str(e)}")

            if stream:
                if return_response_obj:
                    return None, (yield f"Error: {str(e)}")
                else:
                    error_str = str(e)

                    def error_generator() -> Generator[str, None, None]:
                        yield f"Error: {error_str}"

                    return error_generator()
            else:
                return {"error": str(e)}

    def process_tool_calls_and_follow_up(
        self,
        response: Any,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.4,
        max_tokens: int | None = None,
        max_iterations: int = 5,
        cookie: str | None = None,
        stream: bool = True,
    ) -> Generator[str, None, None]:
        """
        Process tool calls recursively and make follow-up requests.

        Args:
            response: Initial response (unused, kept for backward compatibility)
            messages: List of message dictionaries
            model: Model identifier (overrides the default model)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            max_iterations: Maximum number of tool call iterations
            cookie: Airflow cookie for authentication
            stream: Whether to stream the final response

        Returns:
            Generator yielding the final response
        """
        model_to_use = model or self.config.model

        return self.service.process_tool_calls_and_follow_up(
            messages=messages, model=model_to_use, temperature=temperature, max_tokens=max_tokens, cookie=cookie, max_iterations=max_iterations, stream=stream
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "WingmanClient":
        """
        Create a WingmanClient instance from a configuration dictionary.

        Args:
            config: Configuration dictionary with provider_name, api_key,
                   and optional base_url

        Returns:
            WingmanClient instance
        """
        return cls(
            provider_name=config.get("provider_name", "openai"),
            api_key=config.get("api_key", ""),
            base_url=config.get("base_url"),
            model=config.get("model"),
        )
