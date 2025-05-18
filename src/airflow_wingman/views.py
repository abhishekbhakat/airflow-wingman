"""Views for Airflow Wingman plugin."""

import json
import logging

from flask import Response, request, session
from flask.json import jsonify
from flask_appbuilder import BaseView as AppBuilderBaseView, expose

from airflow_wingman.client import WingmanClient
from airflow_wingman.llms_models import MODELS
from airflow_wingman.notes import INTERFACE_MESSAGES
from airflow_wingman.prompt_engineering import prepare_messages
from airflow_wingman.tools import list_airflow_tools

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

            airflow_tools = []
            airflow_cookie = request.cookies.get("session")
            if airflow_cookie:
                try:
                    airflow_tools = list_airflow_tools(airflow_cookie)
                    logger.info(f"Loaded {len(airflow_tools)} Airflow tools")
                    if not len(airflow_tools) > 0:
                        logger.warning("No Airflow tools were loaded")
                except Exception as e:
                    logger.error(f"Error fetching Airflow tools: {str(e)}")

            data["messages"] = prepare_messages(data["messages"])

            provider_name = data.get("provider", "openai")

            base_url = data.get("base_url")

            safe_data = {k: v for k, v in data.items() if k != "api_key"}
            logger.info(f"Chat request: provider={provider_name}, model={data.get('model')}, stream={data.get('stream')}")
            logger.info(f"Request parameters: {json.dumps(safe_data)[:200]}...")

            client = WingmanClient(provider_name=provider_name, api_key=data["api_key"], base_url=base_url)

            client.set_airflow_tools(airflow_tools)

            if data.get("stream", False):
                return self._handle_streaming_response(client, data)
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

        provider = data.get("provider", "openai")
        if provider not in MODELS:
            raise ValueError(f"Unsupported provider: {provider}. Supported providers: {', '.join(MODELS.keys())}")

        return {
            "model": data["model"],
            "messages": data["messages"],
            "api_key": data["api_key"],
            "stream": data.get("stream", False),
            "temperature": data.get("temperature", 0.4),
            "max_tokens": data.get("max_tokens"),
            "cookie": data.get("cookie"),
            "provider": provider,
            "base_url": data.get("base_url"),
        }

    def _handle_streaming_response(self, client: WingmanClient, data: dict) -> Response:
        """Handle streaming response."""
        try:
            logger.info("Beginning streaming response")
            airflow_cookie = request.cookies.get("session")

            streaming_response = client.chat_completion(messages=data["messages"], model=data["model"], temperature=data["temperature"], max_tokens=data["max_tokens"], stream=True)

            def stream_response(cookie=airflow_cookie):
                complete_response = ""

                for chunk in streaming_response:
                    if chunk:
                        complete_response += chunk
                        yield f"data: {chunk}\n\n"

                follow_up_response = client.process_tool_calls_and_follow_up(None, data["messages"], data["model"], data["temperature"], data["max_tokens"], cookie=cookie, stream=True)

                follow_up_complete_response = ""
                for chunk in follow_up_response:
                    if chunk:
                        follow_up_complete_response += chunk
                        yield f"data: {chunk}\n\n"

                if follow_up_complete_response:
                    follow_up_event = json.dumps({"event": "follow_up_response", "content": follow_up_complete_response})
                    yield f"data: {follow_up_event}\n\n"

                complete_event = json.dumps({"event": "complete_response", "content": complete_response})
                yield f"data: {complete_event}\n\n"

                yield "data: [DONE]\n\n"

            return Response(stream_response(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
        except Exception as e:
            logger.error(f"Streaming error: {str(e)}")
            return jsonify({"error": str(e)}), 500

    def _handle_regular_response(self, client: WingmanClient, data: dict) -> Response:
        """Handle regular (non-streaming) response."""
        try:
            logger.info("Beginning regular response")
            airflow_cookie = request.cookies.get("session")
            response = client.chat_completion(messages=data["messages"], model=data["model"], temperature=data["temperature"], max_tokens=data["max_tokens"], stream=False)
            logger.info(f"Regular response received: {response}")

            follow_up_response = client.process_tool_calls_and_follow_up(None, data["messages"], data["model"], data["temperature"], data["max_tokens"], cookie=airflow_cookie, stream=False)

            return jsonify({"response": response, "follow_up_response": follow_up_response})
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            return jsonify({"error": str(e)}), 500
