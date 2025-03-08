"""
Google Gemini provider implementation for Airflow Wingman.

This module contains the Google provider implementation that handles
API requests, tool conversion, and response processing for Google Gemini models.
"""

import json
import logging
import traceback
from typing import Any

from google import genai
from google.genai.types import Content, FunctionDeclaration, GenerateContentConfig, Part, Schema, Tool

from airflow_wingman.providers.base import BaseLLMProvider, StreamingResponse
from airflow_wingman.tools import execute_airflow_tool
from airflow_wingman.tools.conversion import convert_to_google_tools

# Create a properly namespaced logger for the Airflow plugin
logger = logging.getLogger("airflow.plugins.wingman")


class GoogleProvider(BaseLLMProvider):
    """
    Google Gemini provider implementation.

    This class handles API requests, tool conversion, and response processing
    for the Google Gemini API.
    """

    def __init__(self, api_key: str):
        """
        Initialize the Google Gemini provider.

        Args:
            api_key: API key for Google Gemini
        """
        self.api_key = api_key
        self.client = genai.Client(api_key=api_key)

    def convert_tools(self, airflow_tools: list) -> list:
        """
        Convert Airflow tools to Google Gemini format.

        Args:
            airflow_tools: List of Airflow tools from MCP server

        Returns:
            List of Google Gemini tool definitions
        """
        return convert_to_google_tools(airflow_tools)

    def _convert_messages_to_google_format(self, messages: list[dict[str, Any]]) -> list:
        """
        Convert messages from Airflow format to Google Gemini format.

        Args:
            messages: List of message dictionaries with 'role' and 'content'

        Returns:
            List of messages in Google Gemini format
        """
        google_messages = []
        system_message = None

        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")

            # Handle system message separately for Google's API
            if role == "system":
                system_message = content
                continue

            # Map roles from OpenAI to Google format
            google_role = {
                "user": "user",
                "assistant": "model",
                # Tool messages will be handled in create_follow_up_completion
            }.get(role)

            if google_role and content:
                google_messages.append(Content(role=google_role, parts=[Part(text=content)]))

        return google_messages, system_message

    def create_chat_completion(
        self, messages: list[dict[str, Any]], model: str, temperature: float = 0.4, max_tokens: int | None = None, stream: bool = False, tools: list[dict[str, Any]] | None = None
    ) -> Any:
        """
        Make API request to Google Gemini.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model identifier (e.g., "gemini-2.0-flash")
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            tools: List of tool definitions in Google Gemini format

        Returns:
            Google Gemini response object

        Raises:
            Exception: If the API request fails
        """
        has_tools = tools is not None and len(tools) > 0

        try:
            logger.info(f"Sending chat completion request to Google with model: {model}")

            # Convert messages from OpenAI format to Google format
            google_messages, system_message = self._convert_messages_to_google_format(messages)

            # Create the generation config
            config = GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )

            # Add system message if present
            if system_message:
                config.system_instruction = system_message
                logger.info(f"Added system instruction: {system_message[:50]}..." if len(system_message) > 50 else system_message)

            # Add tools if present
            if has_tools:
                # Convert tool dictionaries to proper Tool objects
                tool_objects = self._convert_to_tool_objects(tools)
                config.tools = tool_objects
                logger.info(f"Added {len(tool_objects)} tool objects with {sum(len(t.function_declarations) for t in tool_objects)} functions")
            else:
                logger.warning("No tools included in request")

            # Log request parameters
            request_params = {
                "model": model,
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                "stream": stream,
                "has_tools": has_tools,
                "message_count": len(google_messages),
            }
            logger.info(f"Request parameters: {json.dumps(request_params)}")

            # Make the API request
            try:
                if stream:
                    response = self.client.models.generate_content_stream(model=model, contents=google_messages, config=config)
                else:
                    response = self.client.models.generate_content(model=model, contents=google_messages, config=config)

                logger.info("Received response from Google Gemini")
                return response
            except Exception as api_error:
                error_msg = str(api_error)
                # If the error is related to tools, retry without tools
                if has_tools and ("tools" in error_msg.lower() or "function" in error_msg.lower()):
                    logger.warning(f"Tools-related error: {error_msg}. Retrying without tools...")
                    # Remove tools from config
                    config.tools = None
                    if stream:
                        response = self.client.models.generate_content_stream(model=model, contents=google_messages, config=config)
                    else:
                        response = self.client.models.generate_content(model=model, contents=google_messages, config=config)
                    logger.info("Received response from Google Gemini (retry without tools)")
                    return response
                else:
                    # Re-raise other errors
                    raise
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to get response from Google Gemini: {error_msg}\n{traceback.format_exc()}")
            raise

    def has_tool_calls(self, response: Any) -> bool:
        """
        Check if the response contains tool calls.

        Args:
            response: Google Gemini response object or StreamingResponse with tool_call attribute

        Returns:
            True if the response contains tool calls, False otherwise
        """
        logger.info(f"Checking for tool calls in response of type: {type(response)}")

        # Check if response is a StreamingResponse with a tool_call attribute
        if isinstance(response, StreamingResponse):
            has_tool = response.tool_call is not None
            logger.info(f"Response is a StreamingResponse, has tool_call: {has_tool}")
            # Log the tool call details if present for debugging
            if has_tool:
                try:
                    tool_call_str = json.dumps(response.tool_call)
                    logger.info(f"Tool call in StreamingResponse: {tool_call_str}")
                except Exception as e:
                    logger.warning(f"Could not log tool call details: {str(e)}")
            return has_tool

        # For non-streaming responses
        if hasattr(response, "candidates") and len(response.candidates) > 0:
            logger.info(f"Response has {len(response.candidates)} candidates")
            for i, candidate in enumerate(response.candidates):
                if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                    for j, part in enumerate(candidate.content.parts):
                        if hasattr(part, "function_call") and part.function_call:
                            logger.info(f"Found function call in candidate {i}, part {j}: {part.function_call.name}")
                            return True
        else:
            logger.info("Response has no candidates or empty candidates list")

        logger.info("No tool calls found in response")
        return False

    def get_tool_calls(self, response: Any) -> list:
        """
        Extract tool calls from the response.

        Args:
            response: Google Gemini response object or StreamingResponse with tool_call attribute

        Returns:
            List of tool call objects in a standardized format
        """
        tool_calls = []

        # Check if response is a StreamingResponse with a tool_call attribute
        if isinstance(response, StreamingResponse) and response.tool_call is not None:
            logger.info(f"Extracting tool call from StreamingResponse: {response.tool_call}")
            tool_calls.append(response.tool_call)
            return tool_calls

        # For non-streaming responses
        if hasattr(response, "candidates") and len(response.candidates) > 0:
            logger.info(f"Extracting tool calls from response with {len(response.candidates)} candidates")
            for i, candidate in enumerate(response.candidates):
                if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                    for j, part in enumerate(candidate.content.parts):
                        if hasattr(part, "function_call") and part.function_call:
                            func_call = part.function_call
                            logger.info(f"Found function call in candidate {i}, part {j}: {func_call.name}")

                            # Create a standardized tool call format similar to OpenAI
                            standardized_tool_call = {
                                "id": getattr(func_call, "id", f"call_{len(tool_calls)}"),  # Generate ID if not present
                                "name": func_call.name,
                                "input": func_call.args,  # Note: Google uses args instead of arguments
                            }
                            tool_calls.append(standardized_tool_call)

                            # Log details about the tool call
                            try:
                                args_str = json.dumps(func_call.args)
                                logger.info(f"Tool call details - Name: {func_call.name}, Arguments: {args_str[:100]}..." if len(args_str) > 100 else args_str)
                            except Exception as e:
                                logger.warning(f"Could not log tool call details: {str(e)}")
        else:
            logger.warning("Response has no candidates, cannot extract tool calls")

        logger.info(f"Extracted {len(tool_calls)} tool calls from Google Gemini response")
        return tool_calls

    def process_tool_calls(self, response: Any, cookie: str) -> dict[str, Any]:
        """
        Process tool calls from the response.

        Args:
            response: Google Gemini response object or StreamingResponse with tool_call attribute
            cookie: Airflow cookie for authentication

        Returns:
            Dictionary mapping tool call IDs to results
        """
        results = {}

        if not self.has_tool_calls(response):
            return results

        # Get tool calls using the standardized method
        tool_calls = self.get_tool_calls(response)
        logger.info(f"Processing {len(tool_calls)} tool calls")

        for tool_call in tool_calls:
            tool_id = tool_call["id"]
            function_name = tool_call["name"]
            arguments = tool_call["input"]

            try:
                # Execute the Airflow tool with the provided arguments and cookie
                logger.info(f"Executing tool: {function_name} with arguments: {arguments}")
                result = execute_airflow_tool(function_name, arguments, cookie)
                logger.info(f"Tool execution result: {result}")
                results[tool_id] = {"status": "success", "result": result}
            except Exception as e:
                error_msg = f"Error executing tool: {str(e)}\n{traceback.format_exc()}"
                logger.error(error_msg)
                results[tool_id] = {"status": "error", "message": error_msg}

        return results

    def create_follow_up_completion(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float = 0.4,
        max_tokens: int | None = None,
        tool_results: dict[str, Any] = None,
        original_response: Any = None,
        stream: bool = False,
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        """
        Create a follow-up completion with tool results.

        Args:
            messages: Original messages
            model: Model identifier
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            tool_results: Results of tool executions
            original_response: Original response with tool calls
            stream: Whether to stream the response
            tools: List of tool definitions in Google Gemini format

        Returns:
            Google Gemini response object or StreamingResponse if streaming
        """
        if not original_response or not tool_results:
            return original_response

        # Convert messages to Google format
        google_messages, system_message = self._convert_messages_to_google_format(messages)

        # Handle StreamingResponse objects
        if isinstance(original_response, StreamingResponse):
            logger.info("Processing StreamingResponse in create_follow_up_completion")
            # Extract tool calls from StreamingResponse
            if original_response.tool_call is not None:
                tool_call = original_response.tool_call

                # Create a proper FunctionCall object with 'args' instead of 'input'
                # The Google API expects 'args' but our internal format uses 'input'
                function_call_args = {"name": tool_call["name"], "args": tool_call["input"] if tool_call["input"] else None}
                if "id" in tool_call:
                    function_call_args["id"] = tool_call["id"]

                logger.info(f"Creating function call with args: {function_call_args}")

                # Add assistant response with function call
                assistant_content = Content(role="model", parts=[Part(function_call=function_call_args)])
                google_messages.append(assistant_content)

                # Add tool result as user response
                tool_result = tool_results.get(tool_call["id"], {}).get("result", "")
                user_content = Content(role="user", parts=[Part.from_function_response(name=tool_call["name"], response={"result": tool_result})])
                google_messages.append(user_content)
        else:
            # Handle regular Google Gemini response objects
            logger.info("Processing regular Google Gemini response in create_follow_up_completion")

            # Extract function calls from original response
            tool_calls = self.get_tool_calls(original_response)

            # For each tool call, add an assistant message with the function call
            # and a user message with the function result
            for tool_call in tool_calls:
                # Add assistant response with function call
                assistant_content = Content(role="model", parts=[Part(function_call={"name": tool_call["name"], "args": tool_call["input"]})])
                google_messages.append(assistant_content)

                # Add tool result as user response
                tool_id = tool_call["id"]
                tool_result = tool_results.get(tool_id, {}).get("result", "")
                user_content = Content(role="user", parts=[Part.from_function_response(name=tool_call["name"], response={"result": tool_result})])
                google_messages.append(user_content)

        # Create the generation config for the follow-up request
        config = GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        # Add system message if present
        if system_message:
            config.system_instruction = system_message

        # Add tools if present (for potential follow-up tool calls)
        if tools:
            # Convert tool dictionaries to proper Tool objects
            tool_objects = self._convert_to_tool_objects(tools)
            config.tools = tool_objects
            logger.info(f"Added {len(tool_objects)} tool objects with {sum(len(t.function_declarations) for t in tool_objects)} functions")

        # Make a second request to get the final response
        logger.info(f"Making second request with tool results (stream={stream})")
        # Use the same API call pattern as in create_chat_completion for consistency
        if stream:
            return self.client.models.generate_content_stream(model=model, contents=google_messages, config=config)
        else:
            return self.client.models.generate_content(model=model, contents=google_messages, config=config)

    def get_content(self, response: Any) -> str:
        """
        Extract content from the response.

        Args:
            response: Google Gemini response object

        Returns:
            Content string from the response
        """
        if hasattr(response, "text"):
            return response.text

        if hasattr(response, "candidates") and len(response.candidates) > 0:
            return response.candidates[0].content.parts[0].text

        return ""

    def get_streaming_content(self, response: Any) -> StreamingResponse:
        """
        Get a generator for streaming content from the response.

        Args:
            response: Google Gemini streaming response object

        Returns:
            StreamingResponse object wrapping a generator that yields content chunks
            and can also store tool call information detected during streaming
        """
        logger.info(f"Getting streaming content from Google response of type: {type(response)}")

        # Create the StreamingResponse object first
        streaming_response = StreamingResponse(generator=None, tool_call=None)

        # Track if we've detected a tool call
        tool_use_detected = False
        current_tool_call = None

        def stream_google_response():
            nonlocal tool_use_detected, current_tool_call

            # Flag to track if we've yielded any content
            has_yielded_content = False

            try:
                # Stream tokens from the response
                for chunk in response:
                    logger.debug("Processing streaming chunk")
                    # Check for function calls in the chunk
                    if hasattr(chunk, "candidates") and len(chunk.candidates) > 0:
                        logger.debug(f"Chunk has {len(chunk.candidates)} candidates")
                        for part in chunk.candidates[0].content.parts:
                            # Check for function calls
                            if hasattr(part, "function_call") and part.function_call:
                                func_call = part.function_call

                                # Initialize or update the tool call
                                if not tool_use_detected:
                                    tool_use_detected = True
                                    logger.info(f"Detected function call in stream: {func_call.name}")

                                    # Initialize the tool call
                                    current_tool_call = {
                                        "id": getattr(func_call, "id", "call_1"),  # Generate ID if not present
                                        "name": func_call.name,
                                        "input": func_call.args or {},
                                    }
                                    # Update the StreamingResponse object's tool_call attribute
                                    streaming_response.tool_call = current_tool_call
                                    logger.info(f"Initialized tool call: {current_tool_call['name']}")
                                else:
                                    # Update existing tool call if needed
                                    if func_call.args and current_tool_call:
                                        current_tool_call["input"] = func_call.args
                                        streaming_response.tool_call = current_tool_call
                                        logger.info(f"Updated tool call arguments for: {current_tool_call['name']}")

                                # Log the tool call details
                                try:
                                    if func_call.args:
                                        args_str = json.dumps(func_call.args)
                                        logger.info(f"Tool call details - Name: {func_call.name}, Arguments: {args_str[:100]}..." if len(args_str) > 100 else args_str)
                                except Exception as e:
                                    logger.warning(f"Could not log tool call details: {str(e)}")

                                # Don't yield content for tool calls
                                continue

                            # Get text content if available
                            if hasattr(part, "text") and part.text:
                                yield part.text
                                has_yielded_content = True
                    else:
                        logger.debug("Chunk has no candidates or empty candidates list")

                # If we've detected a tool call but haven't yielded any content,
                # yield a placeholder message so the frontend has something to display
                if tool_use_detected and not has_yielded_content:
                    logger.info("Yielding placeholder content for tool call")
                    yield "I'll help you with that."  # Simple placeholder message

            except Exception as e:
                error_msg = f"Error streaming response: {str(e)}"
                logger.error(f"{error_msg}\n{traceback.format_exc()}")
                yield f"\nError: {error_msg}"

        # Create the generator
        gen = stream_google_response()

        # Set the generator in the StreamingResponse object
        streaming_response.generator = gen

        return streaming_response

    def _convert_to_tool_objects(self, tools: list[dict[str, Any]]) -> list[Tool]:
        """
        Convert dictionary-format tools to Google's Tool objects.

        Args:
            tools: List of tool definitions with function_declarations

        Returns:
            List of Tool objects ready for the Google API
        """
        tool_objects = []
        for tool_dict in tools:
            if "function_declarations" in tool_dict:
                # Extract function declarations from the dictionary
                function_declarations = []
                for func in tool_dict["function_declarations"]:
                    # Create proper FunctionDeclaration objects
                    # Google API requires function parameters schema to be of type OBJECT
                    # If a function has no properties, we need to add a dummy property
                    properties = func["parameters"].get("properties", {})

                    # Special handling for functions with empty properties
                    if not properties:
                        logger.warning(f"Empty properties for function {func['name']}, adding dummy property")
                        # Add a dummy property to satisfy Google API requirements
                        properties = {"_dummy": Schema(type="STRING", description="This is a placeholder parameter")}

                    # Always use OBJECT type for function parameters (Google API requirement)
                    params = Schema(
                        type="OBJECT",  # Function parameters must be OBJECT type
                        properties=properties,
                        required=func["parameters"].get("required", []),
                    )
                    function_declarations.append(FunctionDeclaration(name=func["name"], description=func.get("description", ""), parameters=params))
                # Create a Tool object with the function declarations
                tool_objects.append(Tool(function_declarations=function_declarations))
        return tool_objects
