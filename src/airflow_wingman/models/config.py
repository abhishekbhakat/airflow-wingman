"""
Configuration management for Airflow Wingman.

This module provides Pydantic models for configuration management, ensuring
type safety and validation for all configuration parameters.
"""

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
    def from_dict(cls, config: dict) -> "WingmanConfig":
        """
        Create a WingmanConfig instance from a dictionary.

        Args:
            config: Configuration dictionary

        Returns:
            WingmanConfig instance
        """
        return cls(**config)
