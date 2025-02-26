"""
Conversion utilities for Airflow Wingman tools.

This module contains functions to convert between different tool formats
for various LLM providers (OpenAI, Anthropic, etc.).
"""

from typing import Any


def convert_to_openai_tools(airflow_tools: list) -> list:
    """
    Convert Airflow tools to OpenAI tool definitions.

    Args:
        airflow_tools: List of Airflow tools from MCP server

    Returns:
        List of OpenAI tool definitions
    """
    openai_tools = []

    for tool in airflow_tools:
        # Initialize the OpenAI tool structure
        openai_tool = {"type": "function", "function": {"name": tool.name, "description": tool.description or tool.name, "parameters": {"type": "object", "properties": {}, "required": []}}}

        # Extract parameters directly from inputSchema if available
        if hasattr(tool, "inputSchema") and tool.inputSchema:
            # Set the type and required fields directly from the schema
            if "type" in tool.inputSchema:
                openai_tool["function"]["parameters"]["type"] = tool.inputSchema["type"]

            # Add required parameters if specified
            if "required" in tool.inputSchema:
                openai_tool["function"]["parameters"]["required"] = tool.inputSchema["required"]

            # Add properties from the input schema
            if "properties" in tool.inputSchema:
                for param_name, param_info in tool.inputSchema["properties"].items():
                    # Create parameter definition
                    param_def = {}

                    # Handle different schema constructs
                    if "anyOf" in param_info:
                        _handle_schema_construct(param_def, param_info, "anyOf")
                    elif "oneOf" in param_info:
                        _handle_schema_construct(param_def, param_info, "oneOf")
                    elif "allOf" in param_info:
                        _handle_schema_construct(param_def, param_info, "allOf")
                    elif "type" in param_info:
                        param_def["type"] = param_info["type"]
                        # Add format if available
                        if "format" in param_info:
                            param_def["format"] = param_info["format"]
                    else:
                        param_def["type"] = "string"  # Default type

                    # Add description from title or param name
                    param_def["description"] = param_info.get("description", param_info.get("title", param_name))

                    # Add enum values if available
                    if "enum" in param_info:
                        param_def["enum"] = param_info["enum"]

                    # Add default value if available
                    if "default" in param_info and param_info["default"] is not None:
                        param_def["default"] = param_info["default"]

                    # Add to properties
                    openai_tool["function"]["parameters"]["properties"][param_name] = param_def

        openai_tools.append(openai_tool)

    return openai_tools


def convert_to_anthropic_tools(airflow_tools: list) -> list:
    """
    Convert Airflow tools to Anthropic tool definitions.

    Args:
        airflow_tools: List of Airflow tools from MCP server

    Returns:
        List of Anthropic tool definitions
    """
    anthropic_tools = []

    for tool in airflow_tools:
        # Initialize the Anthropic tool structure
        anthropic_tool = {"name": tool.name, "description": tool.description or tool.name, "input_schema": {}}

        # Extract parameters directly from inputSchema if available
        if hasattr(tool, "inputSchema") and tool.inputSchema:
            # Copy the input schema directly as Anthropic's format is similar to JSON Schema
            anthropic_tool["input_schema"] = tool.inputSchema
        else:
            # Create a minimal schema if none exists
            anthropic_tool["input_schema"] = {"type": "object", "properties": {}, "required": []}

        anthropic_tools.append(anthropic_tool)

    return anthropic_tools


def _handle_schema_construct(param_def: dict[str, Any], param_info: dict[str, Any], construct_type: str) -> None:
    """
    Helper function to handle JSON Schema constructs like anyOf, oneOf, allOf.

    Args:
        param_def: Parameter definition to update
        param_info: Parameter info from the schema
        construct_type: Type of construct (anyOf, oneOf, allOf)
    """
    # Get the list of schemas from the construct
    schemas = param_info[construct_type]

    # Find the first schema with a type
    for schema in schemas:
        if "type" in schema:
            param_def["type"] = schema["type"]

            # Add format if available
            if "format" in schema:
                param_def["format"] = schema["format"]

            # Add enum values if available
            if "enum" in schema:
                param_def["enum"] = schema["enum"]

            # Add default value if available
            if "default" in schema and schema["default"] is not None:
                param_def["default"] = schema["default"]

            break

    # If no type was found, default to string
    if "type" not in param_def:
        param_def["type"] = "string"
