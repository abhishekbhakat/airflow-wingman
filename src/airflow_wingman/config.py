"""
Configuration management for Airflow Wingman.

This module provides Pydantic models for configuration management, ensuring
type safety and validation for all configuration parameters.
"""

import os

from pydantic import BaseModel, Field, field_validator


class WingmanConfig(BaseModel):
    """
    Configuration for Airflow Wingman.

    Provides validated configuration options for the Airflow Wingman plugin,
    including LLM provider settings.
    """

    provider_name: str = Field(
        default="openai",
        description="Name of the LLM provider (openai, anthropic, google)",
    )

    api_key: str = Field(
        default="",
        description="API key for the LLM provider",
    )

    base_url: str | None = Field(
        default=None,
        description="Base URL for the LLM provider API (optional)",
    )

    model: str = Field(
        default="gpt-3.5-turbo-0125",
        description="Default model to use for chat completions",
    )

    temperature: float = Field(
        default=0.4,
        description="Sampling temperature (0-1)",
        ge=0.0,
        le=1.0,
    )

    max_tokens: int | None = Field(
        default=None,
        description="Maximum tokens to generate (optional)",
    )

    @field_validator("provider_name")
    def validate_provider_name(cls, v: str) -> str:
        """Validate that the provider name is supported."""
        supported_providers = ["openai", "anthropic", "google"]

        if v.lower() not in supported_providers:
            raise ValueError(f"Provider '{v}' not supported. Must be one of: {', '.join(supported_providers)}")

        return v.lower()

    @classmethod
    def from_env(cls) -> "WingmanConfig":
        """
        Create a WingmanConfig instance from environment variables.

        Returns:
            WingmanConfig instance
        """
        return cls(
            provider_name=os.environ.get("WINGMAN_PROVIDER", "openai"),
            api_key=os.environ.get("WINGMAN_API_KEY", ""),
            base_url=os.environ.get("WINGMAN_BASE_URL"),
            model=os.environ.get("WINGMAN_MODEL", "gpt-3.5-turbo-0125"),
            temperature=float(os.environ.get("WINGMAN_TEMPERATURE", "0.4")),
            max_tokens=int(os.environ.get("WINGMAN_MAX_TOKENS", "0")) or None,
        )

    @classmethod
    def from_dict(cls, config: dict) -> "WingmanConfig":
        """
        Create a WingmanConfig instance from a dictionary.

        Args:
            config: Configuration dictionary

        Returns:
            WingmanConfig instance
        """
        return cls(**config)
