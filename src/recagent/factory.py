"""Одинаковые настройки ядра для REST и демонстрационного чата."""

from .agent import Agent
from .config.settings import get_settings
from .parsing import OllamaClient
from .providers import DemoProvider
from .recsflow import RecsflowProvider


def build_agent(settings=None, *, mode=None, question_policy="adaptive"):
    config = settings or get_settings()
    provider = RecsflowProvider(config.provider) if config.provider.kind == "recsflow" else DemoProvider()
    return Agent(provider=provider, mode=mode or config.parse_mode,
                 llm=OllamaClient(config.llm.model, config.llm.url, config.llm.timeout_s),
                 max_calls=config.session.max_llm_calls, max_sessions=config.session.max_sessions,
                 ttl=config.session.ttl_s, max_tokens=config.session.max_token_budget,
                 max_questions=config.session.max_clarifications, question_policy=question_policy)
