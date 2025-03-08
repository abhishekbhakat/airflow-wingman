MODELS = {
    "openai": {
        "name": "OpenAI",
        "endpoint": "https://api.openai.com/v1",
        "models": [
            {
                "id": "gpt-4o",
                "name": "GPT-4o",
                "default": True,
                "context_window": 128000,
                "description": "Input $5/M tokens, Output $15/M tokens",
            }
        ],
    },
    "anthropic": {
        "name": "Anthropic",
        "endpoint": "https://api.anthropic.com/v1/messages",
        "models": [
            {
                "id": "claude-3-7-sonnet-20250219",
                "name": "Claude 3.7 Sonnet",
                "default": True,
                "context_window": 200000,
                "description": "Input $3/M tokens, Output $15/M tokens",
            },
            {
                "id": "claude-3-5-haiku-20241022",
                "name": "Claude 3.5 Haiku",
                "default": False,
                "context_window": 200000,
                "description": "Input $0.80/M tokens, Output $4/M tokens",
            },
        ],
    },
    "google": {
        "name": "Google Gemini",
        "endpoint": "https://generativelanguage.googleapis.com/v1beta/generateContent",
        "models": [
            {
                "id": "gemini-2.0-flash",
                "name": "Gemini 2.0 Flash",
                "default": True,
                "context_window": 1000000,
                "description": "Input $0.1/M tokens, Output $0.4/M tokens",
            }
        ],
    },
    "openrouter": {
        "name": "OpenRouter",
        "endpoint": "https://openrouter.ai/api/v1/chat/completions",
        "models": [
            {
                "id": "custom",
                "name": "Custom Model",
                "default": False,
                "context_window": 128000,  # Default context window, will be updated based on model
                "description": "Enter any model name supported by OpenRouter (e.g., 'anthropic/claude-3-opus', 'meta-llama/llama-2-70b')",
            },
        ],
    },
}
