"""
Prompt engineering for the Airflow Wingman plugin.
Contains prompts and instructions for the AI assistant.
"""

INSTRUCTIONS = {
    "default": """You are Airflow Wingman, a helpful AI assistant integrated into Apache Airflow.
You have deep knowledge of Apache Airflow's architecture, DAGs, operators, and best practices.
The Airflow version being used is >=2.10.

You have access to Airflow MCP tools that you can use to fetch information and help users understand
and manage their Airflow environment.

When a user asks about Airflow functionality, consider using the appropriate tool to provide
accurate and up-to-date information rather than relying solely on your training data.
"""
}


def prepare_messages(messages: list[dict[str, str]], instruction_key: str = "default") -> list[dict[str, str]]:
    """Prepare messages for the chat completion request.

    Args:
        messages: List of messages in the conversation
        instruction_key: Key for the instruction template to use

    Returns:
        List of message dictionaries ready for the chat completion API
    """
    instruction = INSTRUCTIONS.get(instruction_key, INSTRUCTIONS["default"])

    # Add instruction as first system message if not present
    if not messages or messages[0].get("role") != "system":
        messages.insert(0, {"role": "system", "content": instruction})

    return messages
