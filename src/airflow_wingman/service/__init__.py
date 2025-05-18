"""
Service layer for Airflow Wingman.

This package contains service classes that encapsulate business logic and
provide a unified interface for interacting with various components.
"""

from airflow_wingman.service.llm_service import LLMService

__all__ = ["LLMService"]
