"""
Tool execution module for Airflow Wingman.

This module contains functions to list and execute Airflow tools.
"""

import asyncio
import json
import logging
import traceback

from airflow import configuration
from airflow_mcp_server.config import AirflowConfig
from airflow_mcp_server.tools.tool_manager import get_airflow_tools, get_tool

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


async def _execute_airflow_tool_async(tool_name: str, arguments: dict, cookie: str) -> str:
    """
    Async implementation to execute an Airflow tool.

    Args:
        tool_name: Name of the tool to execute
        arguments: Arguments to pass to the tool
        cookie: Cookie for authentication

    Returns:
        Result of the tool execution as a string
    """
    try:
        base_url = f"{configuration.conf.get('webserver', 'base_url')}/api/v1/"
        logger.info(f"Setting up AirflowConfig with base_url: {base_url}")

        formatted_cookie = cookie
        if cookie and not cookie.startswith("session="):
            formatted_cookie = f"session={cookie}"
            logger.info(f"Formatted cookie with session prefix: {formatted_cookie[:10]}...")

        config = AirflowConfig(base_url=base_url, cookie=formatted_cookie, auth_token=None)

        logger.info(f"Getting tool: {tool_name}")
        tool = await get_tool(config=config, name=tool_name)

        if not tool:
            error_msg = f"Tool not found: {tool_name}"
            logger.error(error_msg)
            return json.dumps({"error": error_msg})

        logger.info(f"Executing tool: {tool_name} with arguments: {arguments}")

        async with tool.client as client:  # noqa F841
            result = await tool.run(arguments)

        if isinstance(result, dict | list):
            result_str = json.dumps(result, indent=2)
        else:
            result_str = str(result)

        logger.info(f"Tool execution result: {result_str[:100]}...")
        return result_str
    except Exception as e:
        error_msg = f"Error executing tool: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg})


def execute_airflow_tool(tool_name: str, arguments: dict, cookie: str) -> str:
    """
    Synchronous wrapper to execute an Airflow tool.

    Args:
        tool_name: Name of the tool to execute
        arguments: Arguments to pass to the tool
        cookie: Cookie for authentication

    Returns:
        Result of the tool execution as a string
    """
    loop = asyncio.new_event_loop()

    try:
        asyncio.set_event_loop(loop)

        result = loop.run_until_complete(_execute_airflow_tool_async(tool_name, arguments, cookie))
        return result
    except Exception as e:
        error_msg = f"Error in execute_airflow_tool: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg})
    finally:
        loop.close()
