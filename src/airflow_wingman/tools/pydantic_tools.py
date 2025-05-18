"""
Pydantic AI tool conversion for Airflow Wingman.

This module provides a unified tool conversion system using Pydantic AI
to convert between different LLM providers (OpenAI, Anthropic, Google).
"""

import logging
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger("airflow.plugins.wingman")


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


class AirflowTool(BaseModel):
    """Pydantic model for Airflow tools."""

    name: str
    description: str | None = None
    parameters: ToolParameters = Field(default_factory=ToolParameters)


def create_pydantic_tool(airflow_tool: Any) -> AirflowTool:
    """
    Convert an Airflow tool to a Pydantic model.

    Args:
        airflow_tool: Airflow tool from MCP server

    Returns:
        AirflowTool Pydantic model
    """
    tool = AirflowTool(
        name=airflow_tool.name,
        description=airflow_tool.description or airflow_tool.name,
    )

    if hasattr(airflow_tool, "inputSchema") and airflow_tool.inputSchema:
        if "required" in airflow_tool.inputSchema:
            tool.parameters.required = airflow_tool.inputSchema["required"]

        if "properties" in airflow_tool.inputSchema:
            for param_name, param_info in airflow_tool.inputSchema["properties"].items():
                param_type = "string"  # Default type

                if "anyOf" in param_info or "oneOf" in param_info or "allOf" in param_info:
                    construct_type = "anyOf" if "anyOf" in param_info else ("oneOf" if "oneOf" in param_info else "allOf")
                    schemas = param_info[construct_type]

                    for schema in schemas:
                        if "type" in schema:
                            param_type = schema["type"]
                            break
                elif "type" in param_info:
                    param_type = param_info["type"]

                param = ToolParameter(type=param_type, description=param_info.get("description", param_info.get("title", param_name)))

                if "enum" in param_info:
                    param.enum = param_info["enum"]

                if "format" in param_info:
                    param.format = param_info["format"]

                if "default" in param_info and param_info["default"] is not None:
                    param.default = param_info["default"]

                if param.type == "array" and "items" in param_info:
                    param.items = param_info["items"]
                elif param.type == "array":
                    param.items = {"type": "string"}

                tool.parameters.properties[param_name] = param

    return tool


def convert_to_openai_with_pydantic(airflow_tools: list[Any]) -> list[dict[str, Any]]:
    """
    Convert Airflow tools to OpenAI tool definitions using Pydantic AI.

    Args:
        airflow_tools: List of Airflow tools from MCP server

    Returns:
        List of OpenAI tool definitions
    """
    logger.info(f"Converting {len(airflow_tools)} Airflow tools to OpenAI format using Pydantic AI")

    pydantic_tools = []
    for tool in airflow_tools:
        pydantic_tool = create_pydantic_tool(tool)

        openai_tool = {
            "type": "function",
            "function": {
                "name": pydantic_tool.name,
                "description": pydantic_tool.description,
                "parameters": {"type": pydantic_tool.parameters.type, "properties": {}, "required": pydantic_tool.parameters.required},
            },
        }

        for name, param in pydantic_tool.parameters.properties.items():
            param_dict = param.model_dump(exclude_none=True)
            openai_tool["function"]["parameters"]["properties"][name] = param_dict

        pydantic_tools.append(openai_tool)

    logger.info(f"Converted {len(pydantic_tools)} tools to OpenAI format")
    return pydantic_tools


def convert_to_anthropic_with_pydantic(airflow_tools: list[Any]) -> list[dict[str, Any]]:
    """
    Convert Airflow tools to Anthropic tool definitions using Pydantic AI.

    Args:
        airflow_tools: List of Airflow tools from MCP server

    Returns:
        List of Anthropic tool definitions
    """
    logger.info(f"Converting {len(airflow_tools)} Airflow tools to Anthropic format using Pydantic AI")

    pydantic_tools = []
    for tool in airflow_tools:
        pydantic_tool = create_pydantic_tool(tool)

        anthropic_tool = {
            "name": pydantic_tool.name,
            "description": pydantic_tool.description,
            "input_schema": {"type": pydantic_tool.parameters.type, "properties": {}, "required": pydantic_tool.parameters.required},
        }

        for name, param in pydantic_tool.parameters.properties.items():
            param_dict = param.model_dump(exclude_none=True)
            anthropic_tool["input_schema"]["properties"][name] = param_dict

        pydantic_tools.append(anthropic_tool)

    logger.info(f"Converted {len(pydantic_tools)} tools to Anthropic format")
    return pydantic_tools


def convert_to_google_with_pydantic(airflow_tools: list[Any]) -> list[dict[str, Any]]:
    """
    Convert Airflow tools to Google Gemini format using Pydantic AI.

    Args:
        airflow_tools: List of Airflow tools from MCP server

    Returns:
        List of Google Gemini tool definitions wrapped in correct SDK structure
    """
    logger.info(f"Converting {len(airflow_tools)} Airflow tools to Google Gemini format using Pydantic AI")

    function_declarations = []
    for tool in airflow_tools:
        pydantic_tool = create_pydantic_tool(tool)

        function_declaration = {
            "name": pydantic_tool.name,
            "description": pydantic_tool.description,
            "parameters": {"type": pydantic_tool.parameters.type, "properties": {}, "required": pydantic_tool.parameters.required},
        }

        for name, param in pydantic_tool.parameters.properties.items():
            param_dict = param.model_dump(exclude_none=True)
            function_declaration["parameters"]["properties"][name] = param_dict

        function_declarations.append(function_declaration)

    google_tools = [{"function_declarations": function_declarations}]

    logger.info(f"Converted {len(function_declarations)} tools to Google Gemini format with correct SDK structure")
    return google_tools
