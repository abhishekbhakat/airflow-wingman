"""
Tool models for Airflow Wingman.

This module provides Pydantic models for LLM tool definitions and results.
"""

from typing import Any

from pydantic import BaseModel, Field


class ToolParameter(BaseModel):
    """Pydantic model for tool parameters."""

    type: str
    description: str | None = None
    enum: list[str] | None = None
    format: str | None = None
    items: dict[str, Any] | None = None
    default: Any | None = None


class ToolParameters(BaseModel):
    """Pydantic model for tool parameters structure."""

    type: str = "object"
    properties: dict[str, ToolParameter] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)


class ToolDefinition(BaseModel):
    """Pydantic model for tool definitions."""

    name: str
    description: str | None = None
    parameters: ToolParameters = Field(default_factory=ToolParameters)


class ToolResult(BaseModel):
    """Pydantic model for tool execution results."""

    tool_name: str
    tool_args: dict[str, Any]
    result: Any
    error: str | None = None
    status: str = "success"  # success, error

    @property
    def is_success(self) -> bool:
        """Check if the tool execution was successful."""
        return self.status == "success"

    @property
    def is_error(self) -> bool:
        """Check if the tool execution resulted in an error."""
        return self.status == "error"

    @classmethod
    def create_success(cls, tool_name: str, tool_args: dict[str, Any], result: Any) -> "ToolResult":
        """Create a successful tool result."""
        return cls(tool_name=tool_name, tool_args=tool_args, result=result, status="success")

    @classmethod
    def create_error(cls, tool_name: str, tool_args: dict[str, Any], error: str) -> "ToolResult":
        """Create an error tool result."""
        return cls(tool_name=tool_name, tool_args=tool_args, result=None, error=error, status="error")
