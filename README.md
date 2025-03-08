# Airflow Wingman
Airflow plugin to enable LLMs chat in Airflow Webserver.

Internally uses [Airflow MCP Server](https://pypi.org/project/airflow-mcp-server) in safe mode. Only has access to 52 tools which are GET requests as per latest release of Airflow OpenAPI Spec (_i.e. 2.10.0_)


## Usage

Install using pip:

```bash
pip install airflow-wingman
```

## Demo Video

https://github.com/user-attachments/assets/6a459071-dddc-43cb-8e2a-87104d67cf29

## Supported Models

- OpenAI
    - GPT-4o
- Google Gemini
    - Gemini 2.0 Flash
- Anthropic
    - Claude 3.7 Sonnet
    - Claude 3.5 Haiku
- Openrouter
    - Any model but context length is limited to 128K.

## Resources

Pypi: https://pypi.org/project/airflow-wingman
