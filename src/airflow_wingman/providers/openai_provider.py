"""
OpenAI provider implementation for Airflow Wingman.

This module contains the OpenAI provider implementation that handles
API requests, tool conversion, and response processing for OpenAI.
"""

import json
import logging
import traceback
from typing import Any

from openai import OpenAI

from airflow_wingman.providers.base import BaseLLMProvider, StreamingResponse
from airflow_wingman.tools import execute_airflow_tool
from airflow_wingman.tools.conversion import convert_to_openai_tools

# Create a properly namespaced logger for the Airflow plugin
logger = logging.getLogger("airflow.plugins.wingman")


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI provider implementation.

    This class handles API requests, tool conversion, and response processing
    for the OpenAI API. It can also be used for OpenRouter with a custom base URL.
    """

    def __init__(self, api_key: str, base_url: str | None = None):
        """
        Initialize the OpenAI provider.

        Args:
            api_key: API key for OpenAI
            base_url: Optional base URL for the API (used for OpenRouter)
        """
        self.api_key = api_key
        
        # Ensure the base_url doesn't end with /chat/completions to prevent URL duplication
        if base_url and '/chat/completions' in base_url:
            # Strip the /chat/completions part and ensure we have a proper base URL
            base_url = base_url.split('/chat/completions')[0]
            if not base_url.endswith('/v1'):
                base_url = f"{base_url}/v1" if not base_url.endswith('/') else f"{base_url}v1"
            logger.info(f"Modified base_url to prevent endpoint duplication: {base_url}")
            
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def convert_tools(self, airflow_tools: list) -> list:
        """
        Convert Airflow tools to OpenAI format.

        Args:
            airflow_tools: List of Airflow tools from MCP server

        Returns:
            List of OpenAI tool definitions
        """
        return convert_to_openai_tools(airflow_tools)

    def create_chat_completion(
        self, messages: list[dict[str, Any]], model: str, temperature: float = 0.4, max_tokens: int | None = None, stream: bool = False, tools: list[dict[str, Any]] | None = None
    ) -> Any:
        """
        Make API request to OpenAI.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model identifier
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            tools: List of tool definitions in OpenAI format

        Returns:
            OpenAI response object

        Raises:
            Exception: If the API request fails
        """
        # Only include tools if we have any
        has_tools = tools is not None and len(tools) > 0
        tool_choice = "auto" if has_tools else None

        try:
            logger.info(f"Sending chat completion request to OpenAI with model: {model}")

            # Log information about tools
            if not has_tools:
                logger.warning("No tools included in request")

            # Log request parameters
            request_params = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream,
                "tools": tools if has_tools else None,
                "tool_choice": tool_choice,
            }
            logger.info(f"Request parameters: {json.dumps(request_params)}")

            response = self.client.chat.completions.create(
                model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, stream=stream, tools=tools if has_tools else None, tool_choice=tool_choice
            )
            logger.info("Received response from OpenAI")
            return response
        except Exception as e:
            # If the API call fails due to tools not being supported, retry without tools
            error_msg = str(e)
            logger.warning(f"Error in OpenAI API call: {error_msg}")
            if "tools" in error_msg.lower():
                logger.info("Retrying without tools")
                response = self.client.chat.completions.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, stream=stream)
                return response
            else:
                logger.error(f"Failed to get response from OpenAI: {error_msg}\n{traceback.format_exc()}")
                raise

    def has_tool_calls(self, response: Any) -> bool:
        """
        Check if the response contains tool calls.

        Args:
            response: OpenAI response object or StreamingResponse with tool_call attribute

        Returns:
            True if the response contains tool calls, False otherwise
        """
        # Check if response is a StreamingResponse with a tool_call attribute
        if isinstance(response, StreamingResponse) and response.tool_call is not None:
            return True

        # For non-streaming responses
        if hasattr(response, "choices") and len(response.choices) > 0:
            message = response.choices[0].message
            return hasattr(message, "tool_calls") and message.tool_calls

        return False

    def get_tool_calls(self, response: Any) -> list:
        """
        Extract tool calls from the response.

        Args:
            response: OpenAI response object or StreamingResponse with tool_call attribute

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
        if hasattr(response, "choices") and len(response.choices) > 0:
            message = response.choices[0].message
            if hasattr(message, "tool_calls") and message.tool_calls:
                for tool_call in message.tool_calls:
                    standardized_tool_call = {"id": tool_call.id, "name": tool_call.function.name, "input": json.loads(tool_call.function.arguments)}
                    tool_calls.append(standardized_tool_call)

        logger.info(f"Extracted {len(tool_calls)} tool calls from OpenAI response")
        return tool_calls

    def process_tool_calls(self, response: Any, cookie: str) -> dict[str, Any]:
        """
        Process tool calls from the response.

        Args:
            response: OpenAI response object or StreamingResponse with tool_call attribute
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
        self, messages: list[dict[str, Any]], model: str, temperature: float = 0.4, max_tokens: int | None = None, tool_results: dict[str, Any] = None, original_response: Any = None, stream: bool = False, tools: list[dict[str, Any]] | None = None
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
            tools: List of tool definitions in OpenAI format

        Returns:
            OpenAI response object or StreamingResponse if streaming
        """
        if not original_response or not tool_results:
            return original_response

        # Handle StreamingResponse objects
        if isinstance(original_response, StreamingResponse):
            logger.info("Processing StreamingResponse in create_follow_up_completion")
            # Extract tool calls from StreamingResponse
            tool_calls = []
            if original_response.tool_call is not None:
                logger.info(f"Found tool call in StreamingResponse: {original_response.tool_call}")
                tool_call = original_response.tool_call
                # Create a simplified tool call structure for the assistant message
                tool_calls.append({
                    "id": tool_call.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": tool_call.get("name", ""),
                        "arguments": json.dumps(tool_call.get("input", {}))
                    }
                })
            
            # Create a new message with the tool calls
            assistant_message = {
                "role": "assistant",
                "content": None,
                "tool_calls": tool_calls,
            }
        else:
            # Handle regular OpenAI response objects
            logger.info("Processing regular OpenAI response in create_follow_up_completion")
            # Get the original message with tool calls
            original_message = original_response.choices[0].message

            # Create a new message with the tool calls
            assistant_message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in original_message.tool_calls],
            }

        # Create tool result messages
        tool_messages = []
        for tool_call_id, result in tool_results.items():
            tool_messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": result.get("result", str(result))})

        # Add the original messages, assistant message, and tool results
        new_messages = messages + [assistant_message] + tool_messages

        # Make a second request to get the final response
        logger.info(f"Making second request with tool results (stream={stream})")
        return self.create_chat_completion(
            messages=new_messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            tools=tools,  # Pass tools parameter for follow-up
        )

    def get_content(self, response: Any) -> str:
        """
        Extract content from the response.

        Args:
            response: OpenAI response object

        Returns:
            Content string from the response
        """
        return response.choices[0].message.content

    def get_streaming_content(self, response: Any) -> StreamingResponse:
        """
        Get a generator for streaming content from the response.

        Args:
            response: OpenAI streaming response object

        Returns:
            StreamingResponse object wrapping a generator that yields content chunks
            and can also store tool call information detected during streaming
        """
        logger.info("Starting OpenAI streaming response processing")

        # Track only the first tool call detected during streaming
        tool_call = None
        tool_use_detected = False
        current_tool_call = None

        # Create the StreamingResponse object first
        streaming_response = StreamingResponse(generator=None, tool_call=None)

        def generate():
            nonlocal tool_call, tool_use_detected, current_tool_call
            
            # Flag to track if we've yielded any content
            has_yielded_content = False

            for chunk in response:
                # Check for tool call in the delta
                if chunk.choices and hasattr(chunk.choices[0].delta, "tool_calls") and chunk.choices[0].delta.tool_calls:
                    # Tool call detected
                    if not tool_use_detected:
                        tool_use_detected = True
                        logger.info("Tool call detected in streaming response")

                        # Initialize the tool call
                        delta_tool_call = chunk.choices[0].delta.tool_calls[0]
                        current_tool_call = {
                            "id": getattr(delta_tool_call, "id", ""),
                            "name": getattr(delta_tool_call.function, "name", "") if hasattr(delta_tool_call, "function") else "",
                            "input": {},
                        }
                        # Update the StreamingResponse object's tool_call attribute
                        streaming_response.tool_call = current_tool_call
                    else:
                        # Update the existing tool call
                        delta_tool_call = chunk.choices[0].delta.tool_calls[0]

                        # Update the tool call ID if it's provided in this chunk
                        if hasattr(delta_tool_call, "id") and delta_tool_call.id and current_tool_call:
                            current_tool_call["id"] = delta_tool_call.id

                        # Update the function name if it's provided in this chunk
                        if hasattr(delta_tool_call, "function") and hasattr(delta_tool_call.function, "name") and delta_tool_call.function.name and current_tool_call:
                            current_tool_call["name"] = delta_tool_call.function.name

                        # Update the arguments if they're provided in this chunk
                        if hasattr(delta_tool_call, "function") and hasattr(delta_tool_call.function, "arguments") and delta_tool_call.function.arguments and current_tool_call:
                            # Instead of trying to parse each chunk as JSON, accumulate the arguments
                            # and only parse the complete JSON at the end
                            if "_raw_arguments" not in current_tool_call:
                                current_tool_call["_raw_arguments"] = ""
                            
                            # Accumulate the raw arguments
                            current_tool_call["_raw_arguments"] += delta_tool_call.function.arguments
                            
                            # Try to parse the accumulated arguments
                            try:
                                arguments = json.loads(current_tool_call["_raw_arguments"])
                                if isinstance(arguments, dict):
                                    # Successfully parsed the complete JSON
                                    current_tool_call["input"] = arguments  # Replace instead of update
                                    # Update the StreamingResponse object's tool_call attribute
                                    streaming_response.tool_call = current_tool_call
                            except json.JSONDecodeError:
                                # This is expected for partial JSON - we'll try again with the next chunk
                                logger.debug(f"Accumulated partial arguments: {current_tool_call['_raw_arguments']}")

                    # Skip yielding content for tool call chunks
                    continue

                # For the final chunk, set the tool_call attribute
                if chunk.choices and hasattr(chunk.choices[0], "finish_reason") and chunk.choices[0].finish_reason == "tool_calls":
                    logger.info("Streaming response finished with tool_calls reason")
                    
                    # If we haven't yielded any content yet and we're finishing with tool_calls,
                    # yield a placeholder message so the frontend has something to display
                    if not has_yielded_content and tool_use_detected:
                        logger.info("Yielding placeholder content for tool call")
                        yield "I'll help you with that."  # Simple placeholder message
                        has_yielded_content = True
                    if current_tool_call:
                        # One final attempt to parse the arguments if we have accumulated raw arguments
                        if "_raw_arguments" in current_tool_call and current_tool_call["_raw_arguments"]:
                            try:
                                arguments = json.loads(current_tool_call["_raw_arguments"])
                                if isinstance(arguments, dict):
                                    current_tool_call["input"] = arguments
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to parse final arguments: {current_tool_call['_raw_arguments']}")
                                # If we still can't parse it, use an empty dict as fallback
                                if not current_tool_call["input"]:
                                    current_tool_call["input"] = {}
                        
                        # Remove the raw arguments from the final tool call
                        if "_raw_arguments" in current_tool_call:
                            del current_tool_call["_raw_arguments"]
                            
                        tool_call = current_tool_call
                        logger.info(f"Final tool call: {json.dumps(tool_call)}")
                        # Update the StreamingResponse object's tool_call attribute
                        streaming_response.tool_call = tool_call
                    continue

                # Handle regular content chunks
                if chunk.choices and hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    yield content
                    has_yielded_content = True

        # Create the generator
        gen = generate()

        # Set the generator in the StreamingResponse object
        streaming_response.generator = gen

        # Return the StreamingResponse object
        return streaming_response
