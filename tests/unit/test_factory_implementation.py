import pytest

from recagent.config.settings import Settings
from recagent.factory import build_agent, component_identity
from recagent.response_generation import EvidenceResponseGenerator, LLMGroundedResponseGenerator


def test_baseline_component_identity_is_explicit():
    identity = component_identity("baseline-v1")
    assert identity["implementation"] == "baseline-v1"
    assert identity["agent"] == "recagent.agent.Agent"
    assert identity["request_adapter"].endswith("SchemaRequestAdapter")


def test_unknown_implementation_is_rejected():
    with pytest.raises(ValueError, match="Неизвестная реализация"):
        component_identity("future")
    with pytest.raises(ValueError, match="Неизвестная реализация"):
        build_agent(implementation="future")


def test_workflow_v2_has_explicit_migration_identity():
    agent = build_agent(implementation="workflow-v2", mode="rules")
    assert agent.implementation == "workflow-v2"
    assert agent.migration_stage == "validation-state-retrieval"
    assert agent.question_policy == "adaptive"


def test_baseline_keeps_its_previous_question_policy():
    agent = build_agent(implementation="baseline-v1", mode="rules")
    assert agent.question_policy == "none"


def test_response_planner_control_only_changes_optional_generator():
    llm_agent = build_agent(settings=Settings(), mode="ollama")
    deterministic_agent = build_agent(settings=Settings(response={"planner": "deterministic"}), mode="ollama")
    assert isinstance(llm_agent.response_generator, LLMGroundedResponseGenerator)
    assert isinstance(deterministic_agent.response_generator, EvidenceResponseGenerator)
    assert not isinstance(deterministic_agent.response_generator, LLMGroundedResponseGenerator)
