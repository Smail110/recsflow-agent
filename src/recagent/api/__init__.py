"""HTTP API for RecAgent.

`create_app` and `app` are re-exported so that `uvicorn recagent.api:app` and
`from recagent.api import create_app` keep working after the move to src-layout
subpackage organisation.
"""
from __future__ import annotations

from .app import app, create_app
from .errors import ProblemDetail, problem
from .health import HealthResponse

__all__ = ["HealthResponse", "ProblemDetail", "app", "create_app", "problem"]
