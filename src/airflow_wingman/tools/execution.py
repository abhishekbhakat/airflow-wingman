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

# Create a properly namespaced logger for the Airflow plugin
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
        # Set up configuration
        base_url = f"{configuration.conf.get('webserver', 'base_url')}/api/v1/"
        logger.info(f"Setting up AirflowConfig with base_url: {base_url}")

        # Format the cookie properly if it doesn't already have the 'session=' prefix
        formatted_cookie = cookie
        if cookie and not cookie.startswith("session="):
            formatted_cookie = f"session={cookie}"
            logger.info(f"Formatted cookie with session prefix: {formatted_cookie[:10]}...")

        config = AirflowConfig(base_url=base_url, cookie=formatted_cookie, auth_token=None)

        # Get available tools
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
        # Set up configuration
        base_url = f"{configuration.conf.get('webserver', 'base_url')}/api/v1/"
        logger.info(f"Setting up AirflowConfig with base_url: {base_url}")

        # Format the cookie properly if it doesn't already have the 'session=' prefix
        formatted_cookie = cookie
        if cookie and not cookie.startswith("session="):
            formatted_cookie = f"session={cookie}"
            logger.info(f"Formatted cookie with session prefix: {formatted_cookie[:10]}...")

        config = AirflowConfig(base_url=base_url, cookie=formatted_cookie, auth_token=None)

        # Get the tool
        logger.info(f"Getting tool: {tool_name}")
        tool = await get_tool(config=config, name=tool_name)

        if not tool:
            error_msg = f"Tool not found: {tool_name}"
            logger.error(error_msg)
            return json.dumps({"error": error_msg})

        # Execute the tool - ensure the client is in an async context
        logger.info(f"Executing tool: {tool_name} with arguments: {arguments}")

        # The AirflowClient needs to be used as an async context manager
        # to properly initialize its session
        async with tool.client as client:  # noqa F841
            # Now the client has a _session attribute and is in an async context
            result = await tool.run(arguments)

        # Convert result to string
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
    # Create a new event loop for this execution
    # This ensures we're always in a clean async context
    loop = asyncio.new_event_loop()

    try:
        # Set the event loop for this thread
        asyncio.set_event_loop(loop)

        # Run the async function in the new event loop
        result = loop.run_until_complete(_execute_airflow_tool_async(tool_name, arguments, cookie))
        return result
    except Exception as e:
        error_msg = f"Error in execute_airflow_tool: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg})
    finally:
        # Always close the loop to free resources
        loop.close()
