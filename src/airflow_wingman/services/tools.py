"""
Tool execution service for Airflow Wingman.

This module provides services for listing and executing Airflow tools.
"""

import asyncio
import json
import logging
import traceback
from typing import Any

from airflow import configuration
from airflow_mcp_server.config import AirflowConfig
from airflow_mcp_server.tools.tool_manager import get_airflow_tools, get_tool

from airflow_wingman.models.tools import ToolResult

logger = logging.getLogger("airflow.plugins.wingman")


async def _list_airflow_tools_async(cookie: str) -> list:
    """
    Async implementation to list available Airflow tools.

    Args:
        cookie: Cookie for authentication

    Returns:
        List of available Airflow tools
    """
    try:
        base_url = f"{configuration.conf.get('webserver', 'base_url')}/api/v1/"
        logger.info(f"Setting up AirflowConfig with base_url: {base_url}")

        formatted_cookie = cookie
        if cookie and not cookie.startswith("session="):
            formatted_cookie = f"session={cookie}"
            logger.info(f"Formatted cookie with session prefix: {formatted_cookie[:10]}...")

        config = AirflowConfig(base_url=base_url, cookie=formatted_cookie, auth_token=None)

        logger.info("Getting Airflow tools...")
        tools = await get_airflow_tools(config=config, mode="safe")
        logger.info(f"Got {len(tools)} tools")
        return tools
    except Exception as e:
        error_msg = f"Error listing Airflow tools: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return []


def list_airflow_tools(cookie: str) -> list:
    """
    Synchronous wrapper to list available Airflow tools.

    Args:
        cookie: Cookie for authentication

    Returns:
        List of available Airflow tools
    """
    return asyncio.run(_list_airflow_tools_async(cookie))


async def _execute_tool_async(name: str, arguments: dict | str, cookie: str) -> ToolResult:
    """
    Async implementation to execute an Airflow tool.

    Args:
        name: Name of the tool to execute
        arguments: Arguments to pass to the tool (dict or JSON string)
        cookie: Cookie for authentication

    Returns:
        ToolResult object with execution results
    """
    try:
        # Parse arguments if provided as a string
        if isinstance(arguments, str):
            try:
                parsed_arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return ToolResult.create_error(tool_name=name, tool_args={"raw_input": arguments}, error="Invalid JSON arguments")
        else:
            parsed_arguments = arguments

        base_url = f"{configuration.conf.get('webserver', 'base_url')}/api/v1/"
        logger.info(f"Setting up AirflowConfig with base_url: {base_url}")

        formatted_cookie = cookie
        if cookie and not cookie.startswith("session="):
            formatted_cookie = f"session={cookie}"
            logger.info(f"Formatted cookie with session prefix: {formatted_cookie[:10]}...")

        config = AirflowConfig(base_url=base_url, cookie=formatted_cookie, auth_token=None)

        logger.info(f"Getting tool: {name}")
        tool = await get_tool(config=config, name=name)

        if not tool:
            error_msg = f"Tool not found: {name}"
            logger.error(error_msg)
            return ToolResult.create_error(tool_name=name, tool_args=parsed_arguments, error=error_msg)

        logger.info(f"Executing tool: {name} with arguments: {parsed_arguments}")

        async with tool.client as client:  # noqa F841
            result = await tool.run(parsed_arguments)

        if isinstance(result, (dict, list)):
            result_str = json.dumps(result, indent=2)
        else:
            result_str = str(result)

        logger.info(f"Tool execution result: {result_str[:100]}...")

        return ToolResult.create_success(tool_name=name, tool_args=parsed_arguments, result=result)

    except Exception as e:
        error_msg = f"Error executing tool: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)

        return ToolResult.create_error(tool_name=name, tool_args=arguments if isinstance(arguments, dict) else {"raw_input": arguments}, error=str(e))


def execute_tool(name: str, arguments: dict | str, cookie: str) -> Any:
    """
    Synchronous wrapper to execute an Airflow tool.

    Args:
        name: Name of the tool to execute
        arguments: Arguments to pass to the tool (dict or JSON string)
        cookie: Cookie for authentication

    Returns:
        Result of the tool execution
    """
    loop = asyncio.new_event_loop()

    try:
        asyncio.set_event_loop(loop)
        tool_result = loop.run_until_complete(_execute_tool_async(name, arguments, cookie))

        if tool_result.is_success:
            return tool_result.result
        else:
            return {"error": tool_result.error}

    except Exception as e:
        error_msg = f"Error in execute_tool: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return {"error": str(e)}

    finally:
        loop.close()


async def execute_tools_in_parallel(tools: list[dict[str, Any]], cookie: str) -> list[ToolResult]:
    """
    Execute multiple tools in parallel.

    Args:
        tools: List of tool definitions with name and arguments
        cookie: Cookie for authentication

    Returns:
        List of ToolResult objects
    """
    tasks = []
    for tool in tools:
        name = tool.get("name")
        arguments = tool.get("arguments", {})
        if name:
            task = _execute_tool_async(name, arguments, cookie)
            tasks.append(task)

    results = await asyncio.gather(*tasks)
    return results


def execute_multiple_tools(tools: list[dict[str, Any]], cookie: str) -> list[dict[str, Any]]:
    """
    Synchronous wrapper to execute multiple tools in parallel.

    Args:
        tools: List of tool definitions with name and arguments
        cookie: Cookie for authentication

    Returns:
        List of tool execution results
    """
    loop = asyncio.new_event_loop()

    try:
        asyncio.set_event_loop(loop)
        results = loop.run_until_complete(execute_tools_in_parallel(tools, cookie))
        return [result.model_dump() for result in results]

    except Exception as e:
        error_msg = f"Error executing multiple tools: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return [{"error": str(e)}]

    finally:
        loop.close()
