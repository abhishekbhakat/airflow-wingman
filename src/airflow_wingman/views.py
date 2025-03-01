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
            if session.get("airflow_cookie"):
                try:
                    airflow_tools = list_airflow_tools(session["airflow_cookie"])
                except Exception as e:
                    # Log the error but continue without tools
                    print(f"Error fetching Airflow tools: {str(e)}")

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
            "temperature": data.get("temperature", 0.7),
            "max_tokens": data.get("max_tokens"),
            "cookie": data.get("cookie"),
            "provider": provider,
            "base_url": data.get("base_url"),
        }

    def _handle_streaming_response(self, client: LLMClient, data: dict) -> Response:
        """Handle streaming response."""
        try:
            logger.info("Beginning streaming response")
            generator = client.chat_completion(messages=data["messages"], model=data["model"], temperature=data["temperature"], max_tokens=data["max_tokens"], stream=True)

            def stream_response():
                complete_response = ""

                # Send SSE format for each chunk
                for chunk in generator:
                    if chunk:
                        complete_response += chunk
                        yield f"data: {chunk}\n\n"

                # Log the complete assembled response at the end
                logger.info("COMPLETE RESPONSE START >>>")
                logger.info(complete_response)
                logger.info("<<< COMPLETE RESPONSE END")

                # Send the complete response as a special event
                complete_event = json.dumps({"event": "complete_response", "content": complete_response})
                yield f"data: {complete_event}\n\n"

                # Signal the end of the stream
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
