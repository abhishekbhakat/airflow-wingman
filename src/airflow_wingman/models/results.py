"""
Result models for Airflow Wingman.

This module provides Pydantic models for LLM chat messages and results.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class Function(BaseModel):
    """Function call model for chat messages."""

    name: str
    arguments: str


class Message(BaseModel):
    """Base message model for chat completions."""

    role: Literal["system", "user", "assistant", "function", "tool"]
    content: str | None = None
    name: str | None = None
    function_call: Function | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class ChatMessage(BaseModel):
    """Chat message model with additional metadata."""

    role: Literal["system", "user", "assistant", "function", "tool"]
    content: str | None = None
    name: str | None = None
    function_call: Function | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    metadata: dict[str, Any] | None = Field(default_factory=dict)

    @classmethod
    def from_dict(cls, message_dict: dict[str, Any]) -> "ChatMessage":
        """Create a ChatMessage from a dictionary."""
        # Handle plain string content for assistant messages
        if isinstance(message_dict.get("content"), str):
            content = message_dict["content"]
        else:
            content = message_dict.get("content")

        return cls(
            role=message_dict["role"],
            content=content,
            name=message_dict.get("name"),
            function_call=Function(**message_dict["function_call"]) if message_dict.get("function_call") else None,
            tool_calls=message_dict.get("tool_calls"),
            tool_call_id=message_dict.get("tool_call_id"),
            metadata=message_dict.get("metadata", {}),
        )


class ChatResult(BaseModel):
    """Result model for chat completions."""

    id: str
    model: str
    created: int
    messages: list[ChatMessage] = Field(default_factory=list)
    tool_results: list[dict[str, Any]] = Field(default_factory=list)

    def add_message(self, message: ChatMessage | dict[str, Any]) -> None:
        """Add a message to the chat result."""
        if isinstance(message, dict):
            self.messages.append(ChatMessage.from_dict(message))
        else:
            self.messages.append(message)

    def add_tool_result(self, tool_result: dict[str, Any]) -> None:
        """Add a tool result to the chat result."""
        self.tool_results.append(tool_result)
