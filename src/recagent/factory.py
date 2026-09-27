"""Одинаковые настройки ядра для REST и демонстрационного чата."""

from .agent import Agent
from .config.settings import get_settings
from .generic_workflow import GenericWorkflowService
from .observability.catalog_gaps import CatalogGapRecorder
from .parsing import OllamaClient
from .providers import DemoProvider, RecommendationProvider
from .recsflow import RecsflowProvider
from .response_generation import EvidenceResponseGenerator
from .workflow import WorkflowAgent


def build_request_components(backend):
    from .domains.demo import domain_spec, request_adapter
    from .interpretation import LLMRequestInterpreter

    adapter = request_adapter()
    return LLMRequestInterpreter(backend, adapter.descriptor, domain_spec=domain_spec()), adapter


IMPLEMENTATIONS = ("baseline-v1", "workflow-v2")


def component_identity(implementation: str = "workflow-v2") -> dict[str, str]:
    """Return the explicit assembly identity recorded by runners and reports."""
    if implementation not in IMPLEMENTATIONS:
        raise ValueError(f"Неизвестная реализация: {implementation}")
    if implementation == "workflow-v2":
        return {
            "implementation": implementation,
            "status": "integrated-dev",
            "migration_stage": "validation-state-retrieval",
            "validation": "proposal+reducer",
            "provider": "provider+bm25+rrf",
            "renderer": "legacy-grounded",
        }
    return {
        "implementation": implementation,
        "agent": "recagent.agent.Agent",
        "interpreter": "recagent.interpretation.LLMRequestInterpreter",
        "request_adapter": "recagent.request_mapping.SchemaRequestAdapter",
        "provider": "factory-selected RecommendationProvider",
        "renderer": "recagent.response_generation.EvidenceResponseGenerator|LLMGroundedResponseGenerator",
    }


def build_agent(
    settings=None,
    *,
    mode=None,
    question_policy=None,
    implementation="workflow-v2",
    provider: RecommendationProvider | None = None,
    llm=None,
    max_questions=None,
    workflow_config: dict | None = None,
    session_id_factory=None,
    response_generator=None,
):
    if implementation not in IMPLEMENTATIONS:
        raise ValueError(f"Неизвестная реализация: {implementation}")
    config = settings or get_settings()
    selected_provider = provider or (RecsflowProvider(config.provider) if config.provider.kind == "recsflow" else DemoProvider())
    selected_llm = llm or OllamaClient(config.llm.model, config.llm.url, config.llm.timeout_s)
    configured_response_generator = response_generator
    if configured_response_generator is None and config.response.planner == "deterministic":
        # A controlled A/B path: retrieval, grounding and deterministic
        # rendering stay identical; only the optional evidence-selection LLM
        # call is removed. The default remains the historical ``llm`` mode.
        configured_response_generator = EvidenceResponseGenerator()
    agent_cls = WorkflowAgent if implementation == "workflow-v2" else Agent
    agent = agent_cls(
        provider=selected_provider,
        mode=mode or config.parse_mode,
        llm=selected_llm,
        max_calls=config.session.max_llm_calls,
        max_sessions=config.session.max_sessions,
        ttl=config.session.ttl_s,
        max_tokens=config.session.max_token_budget,
        max_questions=max_questions if max_questions is not None else config.session.max_clarifications,
        question_policy=question_policy if question_policy is not None else ("adaptive" if implementation == "workflow-v2" else "none"),
        session_id_factory=session_id_factory,
        response_generator=configured_response_generator,
        catalog_gap_recorder=(
            CatalogGapRecorder(config.observability.catalog_gap_log_path)
            if config.observability.catalog_gap_log_path else None
        ),
    )
    if implementation == "workflow-v2":
        agent.implementation = "workflow-v2"
        agent.migration_stage = "validation-state-retrieval"
        agent.workflow_config = workflow_config or {
            "validation": {"semantic_mode": "code-only", "nli": False},
            "retrieval": {"provider_k": 100, "lexical_k": 100, "fused_k": 150, "dense": False},
            "ranking": {"method": "deterministic-rrf", "rrf_k": 60},
            "reranker": False,
        }
        interpretation = agent.workflow_config.get("interpretation", {})
        agent.set_interpretation_transport(
            interpretation.get("transport", "flat"),
            domain_value_required=interpretation.get("domain_value_required", True),
        )
    return agent


def build_domain_workflow(
    *,
    domain,
    provider,
    backend,
    adapter,
    item_projection=None,
    interpretation_transport="flat",
    domain_value_required=True,
):
    """Assemble a schema-driven customer workflow without demo-domain branches."""
    from .interpretation import LLMRequestInterpreter

    return GenericWorkflowService(
        domain=domain,
        provider=provider,
        interpreter=LLMRequestInterpreter(
            backend,
            adapter.descriptor,
            transport=interpretation_transport,
            domain_spec=domain,
            domain_value_required=domain_value_required,
        ),
        adapter=adapter,
        item_projection=item_projection,
    )
