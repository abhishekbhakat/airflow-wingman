"""Views for Airflow Wingman plugin."""

import json
import logging

from flask import Response, request, session
from flask.json import jsonify
from flask_appbuilder import BaseView as AppBuilderBaseView, expose

from airflow_wingman.llm_client import LLMClient
from airflow_wingman.llms_models import MODELS
from airflow_wingman.notes import INTERFACE_MESSAGES
from airflow_wingman.prompt_engineering import prepare_messages
from airflow_wingman.tools import list_airflow_tools

# Create a properly namespaced logger for the Airflow plugin
logger = logging.getLogger("airflow.plugins.wingman")


class WingmanView(AppBuilderBaseView):
    """View for Airflow Wingman plugin."""

    route_base = "/wingman"
    default_view = "chat"

    @expose("/")
    def chat(self):
        """Render chat interface."""
        providers = {provider: info["name"] for provider, info in MODELS.items()}
        return self.render_template("wingman_chat.html", title="Airflow Wingman", models=MODELS, providers=providers, interface_messages=INTERFACE_MESSAGES)

    @expose("/chat", methods=["POST"])
    def chat_completion(self):
        """Handle chat completion requests."""
        try:
            data = self._validate_chat_request(request.get_json())

            if data.get("cookie"):
                session["airflow_cookie"] = data["cookie"]

            # Get available Airflow tools using the stored cookie
            airflow_tools = []
            airflow_cookie = request.cookies.get("session")
            if airflow_cookie:
                try:
                    airflow_tools = list_airflow_tools(airflow_cookie)
                    logger.info(f"Loaded {len(airflow_tools)} Airflow tools")
                    if not len(airflow_tools) > 0:
                        logger.warning("No Airflow tools were loaded")
                except Exception as e:
                    # Log the error but continue without tools
                    logger.error(f"Error fetching Airflow tools: {str(e)}")

            # Prepare messages with Airflow tools included in the prompt
            data["messages"] = prepare_messages(data["messages"])

            # Get provider name from request or use default
            provider_name = data.get("provider", "openai")

            # Get base URL from models configuration based on provider
            base_url = MODELS.get(provider_name, {}).get("endpoint")

            # Log the request parameters (excluding API key for security)
            safe_data = {k: v for k, v in data.items() if k != "api_key"}
            logger.info(f"Chat request: provider={provider_name}, model={data.get('model')}, stream={data.get('stream')}")
            logger.info(f"Request parameters: {json.dumps(safe_data)[:200]}...")

            # Create a new client for this request with the appropriate provider
            client = LLMClient(provider_name=provider_name, api_key=data["api_key"], base_url=base_url)

            # Set the Airflow tools for the client to use
            client.set_airflow_tools(airflow_tools)

            if data["stream"]:
                return self._handle_streaming_response(client, data)
            else:
                return self._handle_regular_response(client, data)

        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    def _validate_chat_request(self, data: dict) -> dict:
        """Validate chat request data."""
        if not data:
            raise ValueError("No data provided")

        required_fields = ["model", "messages", "api_key"]
        missing = [f for f in required_fields if not data.get(f)]
        if missing:
            raise ValueError(f"Missing required fields: {', '.join(missing)}")

        # Validate provider if provided
        provider = data.get("provider", "openai")
        if provider not in MODELS:
            raise ValueError(f"Unsupported provider: {provider}. Supported providers: {', '.join(MODELS.keys())}")

        return {
            "model": data["model"],
            "messages": data["messages"],
            "api_key": data["api_key"],
            "stream": data.get("stream", True),
            "temperature": data.get("temperature", 0.4),
            "max_tokens": data.get("max_tokens"),
            "cookie": data.get("cookie"),
            "provider": provider,
            "base_url": data.get("base_url"),
        }

    def _handle_streaming_response(self, client: LLMClient, data: dict) -> Response:
        """Handle streaming response."""
        try:
            logger.info("Beginning streaming response")
            # Get the cookie at the beginning of the request handler
            airflow_cookie = request.cookies.get("session")
            logger.info(f"Got airflow_cookie: {airflow_cookie is not None}")

            # Use the enhanced chat_completion method with return_response_obj=True
            streaming_response = client.chat_completion(messages=data["messages"], model=data["model"], temperature=data["temperature"], max_tokens=data["max_tokens"], stream=True)

            def stream_response(cookie=airflow_cookie):
                complete_response = ""

                # Stream the initial response
                for chunk in streaming_response:
                    if chunk:
                        complete_response += chunk
                        yield f"data: {chunk}\n\n"

                # Log the complete assembled response
                logger.info("COMPLETE RESPONSE START >>>")
                logger.info(complete_response)
                logger.info("<<< COMPLETE RESPONSE END")

                # Check for tool calls and make follow-up if needed
                has_tool_calls = client.provider.has_tool_calls(streaming_response)
                logger.info(f"Has tool calls: {has_tool_calls}")
                if has_tool_calls:
                    # Signal tool processing start - frontend should disable send button
                    yield f"data: {json.dumps({'event': 'tool_processing_start'})}\n\n"

                    # Signal to replace content - frontend should clear the current message
                    yield f"data: {json.dumps({'event': 'replace_content'})}\n\n"

                    logger.info("Response contains tool calls, making follow-up request")
                    logger.info(f"Using cookie from closure: {cookie is not None}")

                    # Process tool calls and get follow-up response (handles recursive tool calls)
                    # Always stream the follow-up response for consistent handling
                    follow_up_response = client.process_tool_calls_and_follow_up(
                        streaming_response, data["messages"], data["model"], data["temperature"], data["max_tokens"], cookie=cookie, stream=True
                    )

                    # Collect the follow-up response
                    follow_up_complete_response = ""
                    for chunk in follow_up_response:
                        if chunk:
                            follow_up_complete_response += chunk

                    # Send the follow-up response as a single event
                    if follow_up_complete_response:
                        follow_up_event = json.dumps({"event": "follow_up_response", "content": follow_up_complete_response})
                        logger.info(f"Follow-up event created with length: {len(follow_up_event)}")
                        data_line = f"data: {follow_up_event}\n\n"
                        logger.info(f"Yielding data line with length: {len(data_line)}")
                        yield data_line

                        # Log the complete follow-up response
                        logger.info("FOLLOW-UP RESPONSE START >>>")
                        logger.info(follow_up_complete_response)
                        logger.info("<<< FOLLOW-UP RESPONSE END")

                    # Signal tool processing complete - frontend can re-enable send button
                    yield f"data: {json.dumps({'event': 'tool_processing_complete'})}\n\n"

                # Send the complete response as a special event (for compatibility with existing code)
                complete_event = json.dumps({"event": "complete_response", "content": complete_response})
                yield f"data: {complete_event}\n\n"

                # Signal end of stream
                yield "data: [DONE]\n\n"

            return Response(stream_response(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
        except Exception as e:
            logger.error(f"Streaming error: {str(e)}")
            return jsonify({"error": str(e)}), 500

    def _handle_regular_response(self, client: LLMClient, data: dict) -> Response:
        """Handle regular response."""
        try:
            logger.info("Beginning regular (non-streaming) response")
            response = client.chat_completion(messages=data["messages"], model=data["model"], temperature=data["temperature"], max_tokens=data["max_tokens"], stream=False)
            logger.info("COMPLETE RESPONSE START >>>")
            logger.info(f"Response to frontend: {json.dumps(response)}")
            logger.info("<<< COMPLETE RESPONSE END")

            return jsonify(response)
        except Exception as e:
            logger.error(f"Regular response error: {str(e)}")
            return jsonify({"error": str(e)}), 500
