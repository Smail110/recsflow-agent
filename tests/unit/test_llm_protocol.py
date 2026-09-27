from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace

import httpx
from evals.dataset_v2 import generate_public_dataset
from evals.llm_protocol import (
    JudgeAdvice,
    ProtocolCase,
    SemanticValidation,
    SurfaceReply,
    build_protocol_cases,
    canonical_sha256,
    run_protocol,
)
from evals.llm_replay import OllamaReplayClient, RoleConfig
from evals.oracle import ExpectedOutcome, SpokenConstraints, build_criteria
from evals.scenarios import OracleCriteriaSpec
from scripts import evaluate_llm_protocol as protocol_cli

from recagent.catalog import generate_catalog
from recagent.catalog.users import generate_profiles


def _spec(criteria) -> OracleCriteriaSpec:
    return OracleCriteriaSpec(
        acceptable_ids=sorted(criteria.acceptable_ids),
        ceiling_ids=list(criteria.ceiling_ids),
        threshold=criteria.threshold,
        catalog_size=criteria.catalog_size,
        feasible_count=criteria.feasible_count,
        max_clarifications=criteria.max_clarifications,
    )


def _fixture() -> tuple[ProtocolCase, dict]:
    catalog = generate_catalog(42, 100)
    theta = generate_profiles(42, 1, catalog)[0].theta
    before = SpokenConstraints()
    after = SpokenConstraints(kind="series")
    base = build_criteria(catalog=catalog, theta=theta, user_id="protocol-user", spoken=before, expected=ExpectedOutcome.RECOMMEND)
    branch = build_criteria(catalog=catalog, theta=theta, user_id="protocol-user", spoken=after, expected=ExpectedOutcome.RECOMMEND)
    from evals.llm_protocol import ClarificationBranch

    case = ProtocolCase(
        case_id="dev-protocol-01",
        user_id="protocol-user",
        source_turn_index=0,
        source_case_sha256="a" * 64,
        initial_utterance="Посоветуйте что-нибудь",
        initial_spoken=before,
        base_criteria=_spec(base),
        theta_digest="b" * 64,
        private_theta=theta,
        branches={"kind": ClarificationBranch("kind", "Сериал", {"kind": "series"}, after, _spec(branch))},
    )
    return case, {item.id: item for item in catalog}


@dataclass
class _Recommendation:
    item: object


class _Agent:
    def __init__(self, catalog: dict, *, slot: str = "kind") -> None:
        self.catalog = catalog
        self.slot = slot
        self.calls = []

    def chat(self, request):
        self.calls.append(request)
        if len(self.calls) == 1:
            return SimpleNamespace(
                state="clarify",
                message="Какой формат?",
                clarification_slot=self.slot,
                clarification_slots=[self.slot],
                recommendations=[],
                session_id="random-session-one",
            )
        return SimpleNamespace(
            state="recommend",
            message="Вот варианты",
            clarification_slot=None,
            clarification_slots=[],
            recommendations=[_Recommendation(next(iter(self.catalog.values())))],
            session_id="random-session-two",
        )


class _Client:
    def __init__(self, *, drift: bool = False, judge_error: bool = False) -> None:
        self.drift = drift
        self.judge_error = judge_error

    def structured(self, schema, system_prompt, payload):
        if schema is SurfaceReply:
            return SurfaceReply(message="Сериал", declared_patch=[{"field": "kind", "value": "series"}]), 11
        if schema is SemanticValidation:
            patch = {"kind": "film"} if self.drift else {"kind": "series"}
            return SemanticValidation(valid=True, extracted_patch=[{"field": field, "value": value} for field, value in patch.items()]), 7
        if self.judge_error:
            raise RuntimeError("judge unavailable")
        return JudgeAdvice(
            predicted_state="recommend", grounded=True, public_constraints_satisfied=True, rationale="Проверены видимые факты."
        ), 5


def _run(*, slot: str = "kind", drift: bool = False, judge_error: bool = False):
    case, catalog = _fixture()
    agent = _Agent(catalog, slot=slot)
    result = run_protocol(
        [case],
        catalog_by_id=catalog,
        agent_factory=lambda: agent,
        simulator=_Client(),
        semantic_validator=_Client(drift=drift),
        advisory_judge=_Client(judge_error=judge_error),
    )
    return result, agent


def test_semantic_drift_is_failure_and_does_not_update_spoken() -> None:
    result, agent = _run(drift=True)

    row = result["rows"][0]
    assert result["denominator"] == 1
    assert result["failure_count"] == 1
    assert "simulator_semantic_invalid" in row["failures"]
    assert row["spoken_after_validation"]["kind"] is None
    assert len(agent.calls) == 1


def test_unsupported_actual_slot_stays_in_denominator() -> None:
    result, agent = _run(slot="max_minutes")

    row = result["rows"][0]
    assert result["denominator"] == 1
    assert result["success_count"] == 0
    assert "unsupported_clarification_slot" in row["failures"]
    assert len(agent.calls) == 1


def test_failed_advisory_judge_is_explicit_protocol_failure() -> None:
    result, _ = _run(judge_error=True)

    row = result["rows"][0]
    assert result["denominator"] == 1
    assert result["core_success_count"] in (0, 1)
    assert result["success_count"] == 0
    assert "judge_error" in row["failures"]


def test_replay_hash_ignores_observed_latency_and_session_ids() -> None:
    first, _ = _run()
    second, _ = _run()

    assert first["rows"][0]["spoken_after_validation"]["kind"] == "series"
    assert len(first["rows"][0]["dialogue"]) == 4
    assert first["replay_sha256"] == second["replay_sha256"]
    assert canonical_sha256(first["rows"][0]["observed_latency_ms"]) != ""


def test_short_slate_never_becomes_oracle_success() -> None:
    result, _ = _run()

    row = result["rows"][0]
    assert row["oracle_success"] is False
    assert "short_or_duplicate_slate" in row["failures"]


def test_agent_that_never_clarifies_never_invokes_simulator_roles() -> None:
    case, catalog = _fixture()

    class NoClarificationAgent:
        def chat(self, request):
            return SimpleNamespace(
                state="recommend",
                message="Ответ",
                clarification_slot=None,
                clarification_slots=[],
                recommendations=[_Recommendation(next(iter(catalog.values())))],
                session_id="random",
            )

    class NeverCalled:
        def structured(self, schema, system_prompt, payload):
            raise AssertionError("simulator roles must not be called")

    result = run_protocol(
        [case],
        catalog_by_id=catalog,
        agent_factory=lambda: NoClarificationAgent(),
        simulator=NeverCalled(),
        semantic_validator=NeverCalled(),
        advisory_judge=_Client(),
    )

    row = result["rows"][0]
    assert row["roles"]["simulator"]["attempted"] == 0
    assert row["roles"]["semantic_validator"]["attempted"] == 0
    assert row["roles"]["judge"]["completed"] == 1


def test_branch_keeps_original_threshold_and_only_filters_original_acceptable_ids() -> None:
    cases, _ = generate_public_dataset("dev", per_family=1)
    protocol_cases, _ = build_protocol_cases(cases, generate_catalog(4202), limit=2)
    source = {case.case_id: case for case in cases}

    for protocol_case in protocol_cases:
        original = source[protocol_case.case_id].scenario.criteria
        for branch in protocol_case.branches.values():
            assert branch.criteria.threshold == original.threshold
            assert set(branch.criteria.acceptable_ids) <= set(original.acceptable_ids)


def test_cohort_inclusion_and_exclusion_partition_source_cases() -> None:
    cases, _ = generate_public_dataset("dev", per_family=1)
    for limit in (2, len(cases)):
        selected, manifest = build_protocol_cases(cases, generate_catalog(4202), limit=limit)
        included = {case.case_id for case in selected}
        excluded = set(manifest["excluded_before_inference"])
        assert not included & excluded
        assert included | excluded == {case.case_id for case in cases}
        for case in selected:
            before = case.initial_spoken.model_dump(mode="json")
            for branch in case.branches.values():
                assert all(before[key] is None or before[key] == value for key, value in branch.target_patch.items())


def test_agent_factory_failure_stays_in_denominator() -> None:
    case, catalog = _fixture()

    def broken_factory():
        raise RuntimeError("initialization failed")

    result = run_protocol(
        [case],
        catalog_by_id=catalog,
        agent_factory=broken_factory,
        simulator=_Client(),
        semantic_validator=_Client(),
        advisory_judge=_Client(),
    )
    assert result["denominator"] == 1
    assert result["protocol_complete_count"] == 0
    assert result["success_count"] == 0


def test_agent_followup_failure_marks_protocol_incomplete() -> None:
    case, catalog = _fixture()

    class BrokenFollowup(_Agent):
        def chat(self, request):
            if self.calls:
                raise RuntimeError("followup failed")
            return super().chat(request)

    result = run_protocol(
        [case],
        catalog_by_id=catalog,
        agent_factory=lambda: BrokenFollowup(catalog),
        simulator=_Client(),
        semantic_validator=_Client(),
        advisory_judge=_Client(),
    )
    assert result["denominator"] == 1
    assert result["protocol_complete_count"] == 0
    assert "agent_followup_error:RuntimeError" in result["rows"][0]["failures"]


def test_second_question_counts_even_when_budget_is_exceeded() -> None:
    case, catalog = _fixture()

    class RepeatedQuestion(_Agent):
        def chat(self, request):
            self.calls.clear()
            return super().chat(request)

    result = run_protocol(
        [case],
        catalog_by_id=catalog,
        agent_factory=lambda: RepeatedQuestion(catalog),
        simulator=_Client(),
        semantic_validator=_Client(),
        advisory_judge=_Client(),
    )
    row = result["rows"][0]
    assert row["clarification_counts"]["protocol"] == 2
    assert "clarification_limit_exceeded" in row["failures"]


def test_cache_provenance_excludes_other_identity_and_incomplete(tmp_path) -> None:
    identity = {"model_identity": {"model": "current", "digest": "d"}, "options": {"seed": 42}}
    request = {"role": "judge", "schema": {}, "system_prompt": "p", "input": {}, **identity}
    fingerprint = canonical_sha256({"schema": {}, "system_prompt": "p", "input": {}})
    expected = None
    for directory, model in (("entries", "current"), ("entries", "stale"), ("incomplete", "current")):
        material = {**request, "model_identity": {"model": model, "digest": "d"}}
        key = canonical_sha256(material)
        path = tmp_path / "judge" / directory / f"{key}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"request": material, "cache_key": key, "complete": directory == "entries"}), encoding="utf-8")
        if directory == "entries" and model == "current":
            expected = path.relative_to(tmp_path).as_posix()
    assert set(protocol_cli.cache_entry_hashes(tmp_path, {"judge": {fingerprint}}, {"judge": identity})) == {expected}


def test_cli_persists_incomplete_report_when_identity_setup_fails(tmp_path, monkeypatch) -> None:
    case, _catalog = _fixture()
    output = tmp_path / "result.json"
    manifest = SimpleNamespace(catalog_seed=42, catalog_sha256="catalog-hash", manifest_sha256="manifest-hash")
    monkeypatch.setattr(protocol_cli, "load_dataset", lambda _path: (manifest, []))
    monkeypatch.setattr(protocol_cli, "validate_public_dev_manifest", lambda _value: None)
    monkeypatch.setattr(protocol_cli, "catalog_sha256", lambda _seed: "catalog-hash")
    monkeypatch.setattr(protocol_cli, "generate_catalog", lambda _seed: [])
    frozen = {
        "protocol_version": "v",
        "included_case_ids": [case.case_id],
        "excluded_before_inference": {},
        "cases": [],
        "protocol_sha256": "",
    }
    frozen["protocol_sha256"] = protocol_cli.canonical_sha256({key: value for key, value in frozen.items() if key != "protocol_sha256"})
    monkeypatch.setattr(
        protocol_cli,
        "build_protocol_cases",
        lambda _cases, _catalog, *, limit, max_clarifications: ([case] if limit and max_clarifications else [], frozen),
    )
    monkeypatch.setattr(protocol_cli, "_client", lambda _args, _role: (_ for _ in ()).throw(RuntimeError("Ollama unavailable")))
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_llm_protocol",
            "--manifest",
            "ignored.json",
            "--output",
            str(output),
            "--cache-mode",
            "record",
            "--simulator-model",
            "m",
            "--semantic-validator-model",
            "m",
            "--judge-model",
            "m",
            "--simulator-digest",
            "d",
            "--semantic-validator-digest",
            "d",
            "--judge-digest",
            "d",
        ],
    )

    assert protocol_cli.main() == 2
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "incomplete"
    assert "Ollama unavailable" in report["setup_failure"]
    assert output.with_suffix(".protocol.json").is_file()


def test_llm_replay_record_then_require_cache_for_all_protocol_roles(tmp_path) -> None:
    case, catalog = _fixture()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {"name": "simulator:test", "digest": "sha256:sim"},
                        {"name": "semantic_validator:test", "digest": "sha256:validator"},
                        {"name": "judge:test", "digest": "sha256:judge"},
                    ]
                },
            )
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "test-1"})
        request_schema = json.loads(request.content)["format"]["properties"]
        if "declared_patch" in request_schema:
            content = {"message": "Сериал", "declared_patch": [{"field": "kind", "value": "series"}]}
        elif "extracted_patch" in request_schema:
            content = {
                "valid": True,
                "extracted_patch": [{"field": "kind", "value": "series"}],
                "unsupported_or_conflicting": [],
                "evidence": [],
            }
        else:
            content = {
                "predicted_state": "recommend",
                "grounded": True,
                "public_constraints_satisfied": True,
                "public_constraint_violations": [],
                "evidence_indices": [],
                "insufficient_evidence": False,
                "rationale": "Проверены видимые факты.",
            }
        return httpx.Response(200, json={"message": {"content": json.dumps(content)}, "prompt_eval_count": 3, "eval_count": 2})

    cache_dir = tmp_path / "cache"
    transport = httpx.MockTransport(handler)
    record_clients = {
        role: OllamaReplayClient(
            RoleConfig(role=role, model=f"{role}:test"), cache_dir=cache_dir / role, mode="record", transport=transport
        )
        for role in ("simulator", "semantic_validator", "judge")
    }
    recorded, _ = _run_with_clients(case, catalog, record_clients)
    manifests = cache_dir / "manifests"
    for role, client in record_clients.items():
        client.write_identity_manifest(manifests / f"{role}.json")

    class NoNetwork(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            raise AssertionError(f"require_cache attempted network: {request.url}")

    replay_clients = {
        role: OllamaReplayClient(
            RoleConfig(role=role, model=f"{role}:test"),
            cache_dir=cache_dir / role,
            mode="require_cache",
            frozen_manifest=manifests / f"{role}.json",
            transport=NoNetwork(),
        )
        for role in ("simulator", "semantic_validator", "judge")
    }
    replayed, _ = _run_with_clients(case, catalog, replay_clients)

    assert recorded["replay_sha256"] == replayed["replay_sha256"]
    assert replayed["denominator"] == 1


def _run_with_clients(case, catalog, clients):
    agent = _Agent(catalog)
    result = run_protocol(
        [case],
        catalog_by_id=catalog,
        agent_factory=lambda: agent,
        simulator=clients["simulator"],
        semantic_validator=clients["semantic_validator"],
        advisory_judge=clients["judge"],
    )
    return result, agent


def test_multiturn_accumulates_truth_and_keeps_frozen_oracle_threshold(monkeypatch) -> None:
    from dataclasses import replace

    from evals.llm_protocol import ClarificationBranch

    case, catalog = _fixture()
    genre = next(item.genre for item in catalog.values() if item.kind == "series")
    after_genre = SpokenConstraints(genre=genre)
    case = replace(
        case, branches={**case.branches, "genre": ClarificationBranch("genre", genre, {"genre": genre}, after_genre, case.base_criteria)}
    )

    class TwoQuestions(_Agent):
        def chat(self, request):
            if len(self.calls) == 1:
                self.calls.append(request)
                return SimpleNamespace(
                    state="clarify",
                    message="Какой жанр?",
                    clarification_slot="genre",
                    clarification_slots=["genre"],
                    recommendations=[],
                    session_id="second-session",
                )
            if len(self.calls) == 2:
                self.calls.append(request)
                items = [item for item in catalog.values() if item.kind == "series" and item.genre == genre]
                return SimpleNamespace(
                    state="recommend",
                    message="Подборка",
                    recommendations=[_Recommendation(item) for item in items[:5]],
                    session_id="second-session",
                )
            return super().chat(request)

    class Truthful(_Client):
        def structured(self, schema, system_prompt, payload):
            if schema is SurfaceReply:
                return SurfaceReply(
                    message=payload["canonical_reply"],
                    declared_patch=[{"field": field, "value": value} for field, value in payload["target_patch"].items()],
                ), 11
            if schema is SemanticValidation:
                patch = {"kind": "series"} if payload["requested_slot"] == "kind" else {"genre": genre}
                return SemanticValidation(
                    valid=True, extracted_patch=[{"field": field, "value": value} for field, value in patch.items()]
                ), 7
            return super().structured(schema, system_prompt, payload)

    import evals.llm_protocol as protocol_module

    observed_criteria = []
    original_judge = protocol_module.judge

    def observing_judge(**kwargs):
        observed_criteria.append(kwargs["criteria"])
        return original_judge(**kwargs)

    monkeypatch.setattr(protocol_module, "judge", observing_judge)
    agent = TwoQuestions(catalog)
    result = run_protocol(
        [case],
        catalog_by_id=catalog,
        agent_factory=lambda: agent,
        simulator=Truthful(),
        semantic_validator=Truthful(),
        advisory_judge=Truthful(),
        max_clarifications=3,
    )
    row = result["rows"][0]
    assert row["spoken_after_validation"]["kind"] == "series"
    assert row["spoken_after_validation"]["genre"] == genre
    assert row["clarification_counts"]["protocol"] == 2
    assert row["roles"]["simulator"] == {"attempted": 2, "completed": 2, "tokens": 22}
    assert result["metrics"]["mean_clarifications"] == 2
    assert "clarification_limit_exceeded" not in row["failures"]
    assert agent.calls[-1].session_id == "second-session"
    assert observed_criteria[0].threshold == case.base_criteria.threshold
    assert set(observed_criteria[0].acceptable_ids) <= set(case.base_criteria.acceptable_ids)


def test_independent_claim_audit_detects_false_attribute_despite_positive_judge() -> None:
    case, catalog = _fixture()
    item = next(iter(catalog.values()))

    class FalseExplanation:
        def chat(self, request):
            return SimpleNamespace(
                state="recommend",
                message="Ответ",
                recommendations=[SimpleNamespace(item=item, explanation="Жанр: выдуманный жанр.", evidence=[])],
            )

    result = run_protocol(
        [case],
        catalog_by_id=catalog,
        agent_factory=FalseExplanation,
        simulator=_Client(),
        semantic_validator=_Client(),
        advisory_judge=_Client(),
    )
    assert result["metrics"]["judge_grounded_rate"] == 1
    assert result["metrics"]["unsupported_text_claim_rate"] == 1
    assert result["metrics"]["text_claims"] == 1
    assert result["rows"][0]["explanation_audit"]["unsupported_claims"][0]["claim"] == "Жанр: выдуманный жанр."


def test_no_explanation_claims_are_unknown_rate_not_zero() -> None:
    result, _ = _run()
    assert result["metrics"]["text_claims"] == 0
    assert result["metrics"]["unsupported_text_claim_rate"] is None


def test_cli_factory_defaults_to_main_workflow_and_keeps_explicit_baseline() -> None:
    from recagent.agent import Agent
    from recagent.workflow import WorkflowAgent

    args = protocol_cli.build_parser().parse_args(
        [
            "--manifest",
            "unused.json",
            "--agent-mode",
            "rules",
            "--simulator-model",
            "qwen3:8b",
            "--semantic-validator-model",
            "qwen3:8b",
            "--judge-model",
            "qwen3:8b",
        ]
    )
    assert args.implementation == "workflow-v2"
    assert args.max_clarifications == 3
    agent = protocol_cli.make_agent(args, generate_catalog(42, 100))
    assert isinstance(agent, WorkflowAgent)
    args.implementation = "baseline-v1"
    baseline = protocol_cli.make_agent(args, generate_catalog(42, 100))
    assert type(baseline) is Agent


def test_missing_agent_replay_entry_is_recorded_even_if_product_can_fallback() -> None:
    from evals.llm_replay import CacheMissError

    class MissingReplay:
        def structured(self, schema, system, payload):
            raise CacheMissError("missing pinned agent response")

    backend = protocol_cli.TrackedAgentClient(MissingReplay())
    import pytest

    with pytest.raises(CacheMissError):
        backend.structured(SurfaceReply, "test", {"user": "test"})
    assert len(backend.request_sha256) == 1
    assert backend.errors == ["CacheMissError: missing pinned agent response"]


def test_unknown_role_token_usage_remains_unknown_across_calls() -> None:
    from evals.llm_protocol import _completed_role

    counters = {"attempted": 2, "completed": 0, "tokens": None}
    _completed_role(counters, None)
    _completed_role(counters, 7)
    assert counters["tokens"] is None
    assert counters["completed"] == 2


def test_main_workflow_product_llm_record_replay_is_offline(tmp_path) -> None:
    from dataclasses import replace

    case, catalog = _fixture()
    case = replace(case, initial_utterance="Хочу сериал.", initial_spoken=SpokenConstraints(kind="series"))
    args = protocol_cli.build_parser().parse_args(
        [
            "--manifest",
            "unused.json",
            "--question-policy",
            "none",
            "--simulator-model",
            "qwen3:8b",
            "--semantic-validator-model",
            "qwen3:8b",
            "--judge-model",
            "qwen3:8b",
        ]
    )

    def handler(request):
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "agent:test", "digest": "sha256:agent"}]})
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "test-runtime"})
        body = json.loads(request.content)
        payload = json.loads(body["messages"][-1]["content"])
        if "updates" in body["format"]["properties"]:
            content = {"intent": "discovery", "updates": [{"field": "kind", "value": "series", "source_text": "сериал"}]}
        else:
            content = {"items": [{"item_id": item["id"], "evidence_indexes": [0]} for item in payload["items"][:3]]}
        return httpx.Response(200, json={"message": {"content": json.dumps(content)}, "prompt_eval_count": 10, "eval_count": 5})

    record = OllamaReplayClient(
        RoleConfig(role="agent", model="agent:test"), cache_dir=tmp_path / "agent", mode="record", transport=httpx.MockTransport(handler)
    )
    recorded_backend = protocol_cli.TrackedAgentClient(record)

    def run(backend):
        return run_protocol(
            [case],
            catalog_by_id=catalog,
            agent_factory=lambda: protocol_cli.make_agent(args, list(catalog.values()), backend=backend),
            simulator=_Client(),
            semantic_validator=_Client(),
            advisory_judge=_Client(),
        )

    recorded = run(recorded_backend)
    assert recorded["rows"][0]["dialogue"][-1]["assistant"]["state"] == "recommend"
    assert len(recorded_backend.request_sha256) == 2
    assert recorded_backend.errors == []
    record.write_identity_manifest(tmp_path / "agent-identity.json")

    def no_network(request):
        raise AssertionError(f"Offline replay attempted network: {request.url}")

    replay = OllamaReplayClient(
        RoleConfig(role="agent", model="agent:test"),
        cache_dir=tmp_path / "agent",
        mode="require_cache",
        frozen_manifest=tmp_path / "agent-identity.json",
        transport=httpx.MockTransport(no_network),
    )
    replay_backend = protocol_cli.TrackedAgentClient(replay)
    replayed = run(replay_backend)
    assert replay_backend.errors == []
    assert replay_backend.request_sha256 == recorded_backend.request_sha256
    assert replayed["replay_sha256"] == recorded["replay_sha256"]


def test_workflow_multiturn_record_replay_keeps_source_turn_and_never_falls_back(tmp_path) -> None:
    from dataclasses import replace

    from evals.llm_protocol import ClarificationBranch

    case, catalog = _fixture()
    case = replace(
        case,
        initial_utterance="Нужен курс.",
        initial_spoken=SpokenConstraints(kind="course"),
        branches={
            "genre": ClarificationBranch(
                "genre", "python", {"genre": "python"}, SpokenConstraints(kind="course", genre="python"), case.base_criteria
            )
        },
    )
    args = protocol_cli.build_parser().parse_args(
        [
            "--manifest",
            "unused.json",
            "--question-policy",
            "fixed",
            "--simulator-model",
            "qwen3:8b",
            "--semantic-validator-model",
            "qwen3:8b",
            "--judge-model",
            "qwen3:8b",
        ]
    )
    product_payloads = []

    class TopicClient(_Client):
        def structured(self, schema, system_prompt, payload):
            if schema is SurfaceReply:
                return SurfaceReply(message="python", declared_patch=[{"field": "genre", "value": "python"}]), 5
            if schema is SemanticValidation:
                return SemanticValidation(valid=True, extracted_patch=[{"field": "genre", "value": "python"}]), 5
            return super().structured(schema, system_prompt, payload)

    def handler(request):
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "agent:test", "digest": "sha256:agent"}]})
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "test-runtime"})
        body = json.loads(request.content)
        payload = json.loads(body["messages"][-1]["content"])
        product_payloads.append(payload)
        if "updates" in body["format"]["properties"]:
            update = (
                {"field": "kind", "value": "course", "source_text": "курс"}
                if payload["message"] == "Нужен курс."
                else {"field": "genre", "value": "python", "source_text": "python"}
            )
            content = {"intent": "discovery", "updates": [update]}
        else:
            content = {"items": [{"item_id": item["id"], "evidence_indexes": [0]} for item in payload["items"][:3]]}
        return httpx.Response(200, json={"message": {"content": json.dumps(content)}, "prompt_eval_count": 10, "eval_count": 5})

    agents = []

    def run(backend):
        def factory():
            agent = protocol_cli.make_agent(args, list(catalog.values()), backend=backend)
            agents.append(agent)
            return agent

        return run_protocol(
            [case],
            catalog_by_id=catalog,
            agent_factory=factory,
            simulator=TopicClient(),
            semantic_validator=TopicClient(),
            advisory_judge=TopicClient(),
            max_clarifications=3,
        )

    record = OllamaReplayClient(
        RoleConfig(role="agent", model="agent:test"), cache_dir=tmp_path / "agent", mode="record", transport=httpx.MockTransport(handler)
    )
    recorded_backend = protocol_cli.TrackedAgentClient(record)
    recorded = run(recorded_backend)
    states = [turn["assistant"]["state"] for turn in recorded["rows"][0]["dialogue"] if "assistant" in turn]
    assert states == ["clarify", "recommend"]
    assert len(recorded_backend.request_sha256) == 3
    assert '"source_turn"' in json.dumps(product_payloads[1])
    assert recorded_backend.errors == []
    record.write_identity_manifest(tmp_path / "agent-identity.json")

    def no_network(request):
        raise AssertionError(f"Offline replay attempted network: {request.url}")

    replay = OllamaReplayClient(
        RoleConfig(role="agent", model="agent:test"),
        cache_dir=tmp_path / "agent",
        mode="require_cache",
        frozen_manifest=tmp_path / "agent-identity.json",
        transport=httpx.MockTransport(no_network),
    )
    replay_backend = protocol_cli.TrackedAgentClient(replay)
    replayed = run(replay_backend)
    assert replay_backend.errors == []
    assert replay_backend.request_sha256 == recorded_backend.request_sha256
    assert replayed["replay_sha256"] == recorded["replay_sha256"]
    assert list(agents[0].sessions) == list(agents[1].sessions)
    assert all(agent.sessions[next(iter(agent.sessions))].calls == 3 for agent in agents)


def test_session_id_factory_does_not_overwrite_an_existing_session() -> None:
    import pytest

    from recagent.agent import Agent
    from recagent.models import ChatRequest

    agent = Agent(mode="rules", session_id_factory=lambda: "00000000-0000-0000-0000-000000000001")
    first = agent.chat(ChatRequest(user_id="first", message="Нужен курс Python"))
    with pytest.raises(RuntimeError, match="existing ID"):
        agent.chat(ChatRequest(user_id="second", message="Нужен фильм"))
    assert agent.sessions[str(first.session_id)].user_id == "first"


def test_patch_transport_exposes_declared_properties_and_rejects_unknown_fields() -> None:
    import pytest
    from evals.llm_protocol import SUPPORTED_SLOTS, ReplyFact, patch_values
    from pydantic import TypeAdapter, ValidationError

    fact_type = TypeAdapter(ReplyFact)
    for output, field in [(SurfaceReply, "declared_patch"), (SemanticValidation, "extracted_patch")]:
        schema = output.model_json_schema()
        assert schema["properties"][field]["type"] == "array"
        items = schema["properties"][field]["items"]
        assert items["discriminator"]["propertyName"] == "field"
        names = set()
        for variant in items["oneOf"]:
            fact_schema = schema["$defs"][variant["$ref"].split("/")[-1]]
            assert set(fact_schema["properties"]) == {"field", "value"}
            names.add(fact_schema["properties"]["field"]["const"])
            assert fact_schema["additionalProperties"] is False
        assert names == SUPPORTED_SLOTS
    with pytest.raises(ValidationError):
        fact_type.validate_python({"field": "price", "value": "1200"})
    with pytest.raises(ValidationError):
        fact_type.validate_python({"field": "kind", "value": "course", "extra": "value"})
    with pytest.raises(ValidationError):
        fact_type.validate_python({"field": "practical", "value": "false"})
    with pytest.raises(ValidationError):
        SemanticValidation(valid=True, extracted_patch=[{"field": "kind", "value": "film"}, {"field": "kind", "value": "course"}])
    assert patch_values([fact_type.validate_python({"field": "practical", "value": False})]) == {"practical": False}
    assert patch_values([]) == {}


def test_typed_empty_patch_still_fails_frozen_nonempty_semantic_gate() -> None:
    case, catalog = _fixture()

    class EmptyValidator(_Client):
        def structured(self, schema, system_prompt, payload):
            if schema is SemanticValidation:
                return SemanticValidation(valid=True, extracted_patch=[]), 5
            return super().structured(schema, system_prompt, payload)

    agent = _Agent(catalog)
    result = run_protocol(
        [case],
        catalog_by_id=catalog,
        agent_factory=lambda: agent,
        simulator=_Client(),
        semantic_validator=EmptyValidator(),
        advisory_judge=_Client(),
    )
    assert "simulator_semantic_invalid" in result["rows"][0]["failures"]
    assert result["rows"][0]["spoken_after_validation"]["kind"] is None
    assert len(agent.calls) == 1


def test_patch_fields_must_be_present_but_can_explicitly_be_empty() -> None:
    import pytest
    from pydantic import ValidationError

    for schema, field, payload in [
        (SurfaceReply, "declared_patch", {"message": "Не имеет значения"}),
        (SemanticValidation, "extracted_patch", {"valid": True, "evidence": ["Не имеет значения"]}),
    ]:
        assert field in schema.model_json_schema()["required"]
        with pytest.raises(ValidationError) as failure:
            schema.model_validate(payload)
        assert any(error["type"] == "missing" and error["loc"] == (field,) for error in failure.value.errors())
        explicit = schema.model_validate({**payload, field: []})
        assert getattr(explicit, field) == []


def test_omitted_validator_patch_is_incomplete_not_invented_empty_fact_list() -> None:
    case, catalog = _fixture()

    class IncompleteValidator(_Client):
        def structured(self, schema, system_prompt, payload):
            if schema is SemanticValidation:
                return {"valid": True, "evidence": [payload["message"]]}, 5
            return super().structured(schema, system_prompt, payload)

    agent = _Agent(catalog)
    result = run_protocol(
        [case],
        catalog_by_id=catalog,
        agent_factory=lambda: agent,
        simulator=_Client(),
        semantic_validator=IncompleteValidator(),
        advisory_judge=_Client(),
    )
    row = result["rows"][0]
    assert "semantic_validator_error" in row["failures"]
    assert row["roles"]["semantic_validator"]["completed"] == 0
    assert row["spoken_after_validation"]["kind"] is None
    assert result["protocol_complete_count"] == 0
    assert len(agent.calls) == 1


def test_fact_values_follow_declared_domain_enums_without_target_based_correction() -> None:
    import pytest
    from evals.llm_protocol import ReplyFact, patch_values
    from pydantic import TypeAdapter, ValidationError

    adapter = TypeAdapter(ReplyFact)
    for field, value in [("kind", "series"), ("genre", "python"), ("tone", "мрачный"), ("level", "начальный"), ("practical", False)]:
        fact = adapter.validate_python({"field": field, "value": value})
        assert patch_values([fact]) == {field: value}
    for field, value in [
        ("kind", "сериал"),
        ("kind", "python"),
        ("genre", "course"),
        ("tone", "начальный"),
        ("level", "мрачный"),
        ("practical", "false"),
    ]:
        with pytest.raises(ValidationError):
            adapter.validate_python({"field": field, "value": value})
    unknown = SemanticValidation(valid=False, extracted_patch=[], unsupported_or_conflicting=["unknown user preference"])
    assert unknown.extracted_patch == [] and unknown.valid is False
