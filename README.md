# Airflow Wingman
Airflow plugin to enable LLMs chat in Airflow Webserver.

Internally uses [Airflow MCP Server](https://pypi.org/project/airflow-mcp-server) in safe mode. Only has access to 52 tools which are GET requests as per latest release of Airflow OpenAPI Spec (_i.e. 2.10.0_)


## Usage

Install using pip:

```bash
pip install airflow-wingman
```
