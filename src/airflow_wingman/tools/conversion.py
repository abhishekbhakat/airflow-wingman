"""
Conversion utilities for Airflow Wingman tools.

This module contains functions to convert between different tool formats
for various LLM providers (OpenAI, Anthropic, etc.).
"""

import logging
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
        openai_tool = {"type": "function", "function": {"name": tool.name, "description": tool.description or tool.name, "parameters": {"type": "object", "properties": {}, "required": []}}}

        if hasattr(tool, "inputSchema") and tool.inputSchema:
            if "type" in tool.inputSchema:
                openai_tool["function"]["parameters"]["type"] = tool.inputSchema["type"]

            if "required" in tool.inputSchema:
                openai_tool["function"]["parameters"]["required"] = tool.inputSchema["required"]

            if "properties" in tool.inputSchema:
                for param_name, param_info in tool.inputSchema["properties"].items():
                    param_def = {}

                    if "anyOf" in param_info:
                        _handle_schema_construct(param_def, param_info, "anyOf")
                    elif "oneOf" in param_info:
                        _handle_schema_construct(param_def, param_info, "oneOf")
                    elif "allOf" in param_info:
                        _handle_schema_construct(param_def, param_info, "allOf")
                    elif "type" in param_info:
                        param_def["type"] = param_info["type"]
                        if "format" in param_info:
                            param_def["format"] = param_info["format"]
                    else:
                        param_def["type"] = "string"  # Default type

                    param_def["description"] = param_info.get("description", param_info.get("title", param_name))

                    if "enum" in param_info:
                        param_def["enum"] = param_info["enum"]

                    if "default" in param_info and param_info["default"] is not None:
                        param_def["default"] = param_info["default"]

                    if param_def.get("type") == "array" and "items" not in param_def:
                        if "items" in param_info:
                            param_def["items"] = param_info["items"]
                        else:
                            param_def["items"] = {"type": "string"}

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
    logger = logging.getLogger("airflow.plugins.wingman")
    logger.info(f"Converting {len(airflow_tools)} Airflow tools to Anthropic format")
    anthropic_tools = []

    for tool in airflow_tools:
        anthropic_tool = {"name": tool.name, "description": tool.description or tool.name, "input_schema": {}}

        if hasattr(tool, "inputSchema") and tool.inputSchema:
            anthropic_tool["input_schema"] = tool.inputSchema
        else:
            anthropic_tool["input_schema"] = {"type": "object", "properties": {}, "required": []}

        anthropic_tools.append(anthropic_tool)

    logger.info(f"Converted {len(anthropic_tools)} tools to Anthropic format")
    return anthropic_tools


def convert_to_google_tools(airflow_tools: list) -> list:
    """
    Convert Airflow tools to Google Gemini format.

    Args:
        airflow_tools: List of Airflow tools from MCP server

    Returns:
        List of Google Gemini tool definitions wrapped in correct SDK structure
    """
    logger = logging.getLogger("airflow.plugins.wingman")
    logger.info(f"Converting {len(airflow_tools)} Airflow tools to Google Gemini format")

    function_declarations = []

    for tool in airflow_tools:
        function_declaration = {
            "name": tool.name if hasattr(tool, "name") else str(tool),
            "description": tool.description if hasattr(tool, "description") else "",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }

        if hasattr(tool, "inputSchema") and tool.inputSchema:
            if "required" in tool.inputSchema:
                function_declaration["parameters"]["required"] = tool.inputSchema["required"]

            if "properties" in tool.inputSchema:
                for param_name, param_info in tool.inputSchema["properties"].items():
                    param_def = {}

                    if "anyOf" in param_info:
                        _handle_schema_construct(param_def, param_info, "anyOf")
                    elif "oneOf" in param_info:
                        _handle_schema_construct(param_def, param_info, "oneOf")
                    elif "allOf" in param_info:
                        _handle_schema_construct(param_def, param_info, "allOf")
                    elif "type" in param_info:
                        param_def["type"] = param_info["type"]
                        if "format" in param_info:
                            param_def["format"] = param_info["format"]
                    else:
                        param_def["type"] = "string"  # Default type

                    param_def["description"] = param_info.get("description", param_info.get("title", param_name))

                    if "enum" in param_info:
                        param_def["enum"] = param_info["enum"]

                    if param_def.get("type") == "array" and "items" not in param_def:
                        if "items" in param_info:
                            param_def["items"] = param_info["items"]
                        else:
                            param_def["items"] = {"type": "string"}

                    function_declaration["parameters"]["properties"][param_name] = param_def

        function_declarations.append(function_declaration)

    google_tools = [{"function_declarations": function_declarations}]

    logger.info(f"Converted {len(function_declarations)} tools to Google Gemini format with correct SDK structure")
    return google_tools


def _handle_schema_construct(param_def: dict[str, Any], param_info: dict[str, Any], construct_type: str) -> None:
    """
    Helper function to handle JSON Schema constructs like anyOf, oneOf, allOf.

    Args:
        param_def: Parameter definition to update
        param_info: Parameter info from the schema
        construct_type: Type of construct (anyOf, oneOf, allOf)
    """
    schemas = param_info[construct_type]

    for schema in schemas:
        if "type" in schema:
            param_def["type"] = schema["type"]

            if "format" in schema:
                param_def["format"] = schema["format"]

            if "enum" in schema:
                param_def["enum"] = schema["enum"]

            if "default" in schema and schema["default"] is not None:
                param_def["default"] = schema["default"]

            break

    if "type" not in param_def:
        param_def["type"] = "string"

    if param_def.get("type") == "array" and "items" not in param_def:
        param_def["items"] = {"type": "string"}
