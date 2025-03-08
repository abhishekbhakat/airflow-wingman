"""
Provider factory for Airflow Wingman.

This module contains the factory function to create provider instances
based on the provider name.
"""

from airflow_wingman.providers.anthropic_provider import AnthropicProvider
from airflow_wingman.providers.base import BaseLLMProvider
from airflow_wingman.providers.google_provider import GoogleProvider
from airflow_wingman.providers.openai_provider import OpenAIProvider


def create_llm_provider(provider_name: str, api_key: str, base_url: str | None = None) -> BaseLLMProvider:
    """
    Create a provider instance based on the provider name.

    Args:
        provider_name: Name of the provider (openai, anthropic, openrouter, google)
        api_key: API key for the provider
        base_url: Optional base URL for the provider API

    Returns:
        Provider instance

    Raises:
        ValueError: If the provider is not supported
    """
    provider_name = provider_name.lower()

    if provider_name == "openai":
        return OpenAIProvider(api_key=api_key, base_url=base_url)
    elif provider_name == "openrouter":
        # OpenRouter uses the OpenAI API format, so we can use the OpenAI provider
        # with a custom base URL
        if not base_url:
            base_url = "https://openrouter.ai/api/v1"
        return OpenAIProvider(api_key=api_key, base_url=base_url)
    elif provider_name == "anthropic":
        return AnthropicProvider(api_key=api_key)
    elif provider_name == "google":
        return GoogleProvider(api_key=api_key)
    else:
        raise ValueError(f"Unsupported provider: {provider_name}. Supported providers: openai, anthropic, openrouter, google")
