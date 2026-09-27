from recagent.interpretation import ConstraintUpdate, StructuredRequest
from recagent.models import ChatRequest, Item
from recagent.providers import DemoProvider
from recagent.query_compilation import compile_constraints
from recagent.workflow import WorkflowAgent


class Interpreter:
    def __init__(self):
        self.calls = 0

    def interpret(self, message, previous, **context):
        self.calls += 1
        if self.calls == 1:
            return StructuredRequest(
                updates=[
                    ConstraintUpdate(field="kind", value="course", source_text="курс"),
                    ConstraintUpdate(field="level", value="начальный", source_text="middle"),
                ]
            ), 1
        assert context["unresolved"]
        return StructuredRequest(
            updates=[
                ConstraintUpdate(field="kind", value="course", source_text="курс"),
                ConstraintUpdate(field="genre", value="python", source_text="Python"),
                ConstraintUpdate(field="level", value="продвинутый", source_text="продвинутый"),
            ]
        ), 1


def test_v2_keeps_active_state_unchanged_until_full_proposal_is_valid():
    agent = WorkflowAgent(mode="ollama", interpreter=Interpreter())
    first = agent.chat(ChatRequest(user_id="v2", message="Нужен курс уровня middle"))
    assert first.state == "clarify"
    assert first.query.kind is None
    state = agent.sessions[first.session_id].constraint_state
    assert state.constraints == () and state.pending is not None

    second = agent.chat(ChatRequest(user_id="v2", session_id=first.session_id, message="Нужен курс Python для продвинутый"))
    assert second.state == "recommend"
    assert second.query.kind == "course"
    assert second.query.level == "продвинутый"
    assert agent.sessions[first.session_id].constraint_state.pending is None


def test_v2_uses_declared_public_quality_after_hard_filtering():
    class OrderedProvider(DemoProvider):
        def retrieve(self, user_id, query, limit=100):
            del user_id, query, limit
            return ["b", "a"]

    provider = OrderedProvider(
        items=[
            Item(id="a", title="Лёгкий сериал A", kind="series", tone="лёгкий", quality=0.1, description="Лёгкий сериал"),
            Item(id="b", title="Лёгкий сериал B", kind="series", tone="лёгкий", quality=0.99, description="Лёгкий сериал"),
        ]
    )
    result = WorkflowAgent(provider=provider, mode="rules").chat(ChatRequest(user_id="rrf", message="Лёгкий сериал"))
    # The accepted public-quality prior precedes RRF inside an equal-affinity
    # group. This invented-catalog integration check is not a DEV label.
    assert [recommendation.item.id for recommendation in result.recommendations[:2]] == ["b", "a"]
    assert all(recommendation.item.kind == "series" and recommendation.item.tone == "лёгкий" for recommendation in result.recommendations)
    assert "rrf" in result.trace


def test_v2_refuses_undeclared_provider_capabilities_before_filtering():
    class UndeclaredProvider(DemoProvider):
        def __init__(self):
            super().__init__()
            self.capabilities = None

    response = WorkflowAgent(provider=UndeclaredProvider(), mode="rules").chat(
        ChatRequest(user_id="capability", message="Лёгкий детективный сериал")
    )
    assert response.state == "clarify"
    assert "не объявил" in response.message


def test_v2_does_not_use_an_unverified_title_lookup_for_navigation():
    class UnverifiedTitles(DemoProvider):
        title_lookup_verified = False

    response = WorkflowAgent(provider=UnverifiedTitles(), mode="rules").chat(
        ChatRequest(user_id="title", message="Найди «Декоратор в проде»")
    )
    assert response.state == "clarify"
    assert "не подтверждает уникальность" in response.message


def test_v2_uses_exact_title_lookup_while_baseline_keeps_typo_tolerance():
    provider = DemoProvider()
    exact = next(item for item in provider.items.values() if item.title == "Декоратор в проде")
    assert provider.find_title("Декоратор в прод") == exact
    assert provider.find_title_verified("Декоратор в прод") is None

    response = WorkflowAgent(provider=provider, mode="rules").chat(ChatRequest(user_id="exact-title", message="Найди «Декоратор в прод»"))
    assert response.state == "clarify"
    assert "нет «Декоратор в прод»" in response.message


def test_v2_treats_a_verified_seed_title_as_routing_metadata_not_a_hard_filter():
    agent = WorkflowAgent(provider=DemoProvider(), mode="rules")
    response = agent.chat(ChatRequest(user_id="verified-title", message="Найди «Декоратор в проде»"))
    assert response.state == "recommend"
    assert [recommendation.item.id for recommendation in response.recommendations] == ["it-00022"]


def test_v2_keeps_missing_quoted_title_as_navigation_when_title_contains_catalog_words():
    class NavigationInterpreter:
        def interpret(self, message, previous, **context):
            del message, previous, context
            return StructuredRequest(
                intent="navigation",
                updates=[
                    ConstraintUpdate(
                        field="seed_title",
                        value="Несуществующий курс по Python",
                        source_text="«Несуществующий курс по Python»",
                    )
                ],
            ), 1

    response = WorkflowAgent(mode="ollama", interpreter=NavigationInterpreter()).chat(
        ChatRequest(user_id="missing-title", message="Найди «Несуществующий курс по Python»")
    )
    assert response.state == "clarify"
    assert response.query.intent == "navigation"
    assert response.query.seed_title == "Несуществующий курс по Python"
    assert "нет «Несуществующий курс по Python»" in response.message


def test_v2_applies_declared_provider_and_lexical_windows():
    class WindowProvider(DemoProvider):
        received_limit = None

        def retrieve(self, user_id, query, limit=100):
            del user_id, query
            self.received_limit = limit
            return ["a", "b", "c"]

    items = [
        Item(id=item_id, title=f"Сериал {item_id}", kind="series", tone="лёгкий", quality=0.5, description="лёгкий сериал")
        for item_id in ("a", "b", "c")
    ]
    agent = WorkflowAgent(provider=WindowProvider(items=items), mode="rules")
    agent.workflow_config = {
        "validation": {"semantic_mode": "code-only"},
        "retrieval": {"provider_k": 1, "lexical_k": 1, "fused_k": 1, "dense": False},
        "ranking": {"method": "deterministic-rrf", "rrf_k": 60},
        "reranker": False,
    }
    response = agent.chat(ChatRequest(user_id="window", message="Лёгкий сериал"))
    assert agent.provider.received_limit == 1
    assert [recommendation.item.id for recommendation in response.recommendations] == ["a"]


def test_v2_keeps_tone_exclusion_canonical_and_filters_unknown_metadata():
    class ToneInterpreter:
        def interpret(self, message, previous, **context):
            del message, previous, context
            return StructuredRequest(
                updates=[
                    ConstraintUpdate(field="kind", value="film", source_text="\u0444\u0438\u043b\u044c\u043c"),
                    ConstraintUpdate(
                        field="tone",
                        operation="exclude",
                        value="\u043c\u0440\u0430\u0447\u043d\u044b\u0439",
                        source_text="\u043d\u0435 \u043c\u0440\u0430\u0447\u043d\u044b\u0439",
                    ),
                ]
            ), 1

    provider = DemoProvider(
        items=[
            Item(
                id="dark",
                title="\u0422\u0451\u043c\u043d\u044b\u0439",
                kind="film",
                tone="\u043c\u0440\u0430\u0447\u043d\u044b\u0439",
                quality=0.9,
                description="\u0444\u0438\u043b\u044c\u043c",
            ),
            Item(
                id="light",
                title="\u0421\u0432\u0435\u0442\u043b\u044b\u0439",
                kind="film",
                tone="\u043b\u0451\u0433\u043a\u0438\u0439",
                quality=0.8,
                description="\u0444\u0438\u043b\u044c\u043c",
            ),
            Item(
                id="unknown",
                title="\u0411\u0435\u0437 \u0442\u043e\u043d\u0430",
                kind="film",
                tone=None,
                quality=1.0,
                description="\u0444\u0438\u043b\u044c\u043c",
            ),
        ]
    )
    agent = WorkflowAgent(provider=provider, mode="ollama", interpreter=ToneInterpreter())
    response = agent.chat(
        ChatRequest(
            user_id="tone",
            message="\u041d\u0443\u0436\u0435\u043d \u0444\u0438\u043b\u044c\u043c \u043d\u0435 \u043c\u0440\u0430\u0447\u043d\u044b\u0439",
        )
    )
    constraints = agent.sessions[response.session_id].constraint_state.constraints
    assert response.state == "recommend"
    assert response.query.tone is None  # legacy projection never invents light
    assert {(c.field, c.op, c.value) for c in constraints} >= {("tone", "neq", "\u043c\u0440\u0430\u0447\u043d\u044b\u0439")}
    assert [rec.item.id for rec in response.recommendations] == ["light"]
    compiled = compile_constraints(constraints, provider.capabilities)
    assert any(c.field == "tone" and c.op == "neq" for c in compiled.residual_constraints)


def test_v2_include_removes_only_matching_exclusion_without_creating_positive_tone():
    class OperationsInterpreter:
        def __init__(self):
            self.calls = 0

        def interpret(self, message, previous, **context):
            del message, previous, context
            self.calls += 1
            if self.calls == 1:
                return StructuredRequest(
                    updates=[
                        ConstraintUpdate(field="kind", value="film", source_text="фильм"),
                        ConstraintUpdate(field="tone", operation="exclude", value="мрачный", source_text="без мрачного"),
                    ]
                ), 1
            return StructuredRequest(
                updates=[ConstraintUpdate(field="tone", operation="include", value="мрачный", source_text="мрачный")]
            ), 1

    agent = WorkflowAgent(mode="ollama", interpreter=OperationsInterpreter())
    first = agent.chat(ChatRequest(user_id="include", message="Нужен фильм без мрачного"))
    second = agent.chat(ChatRequest(user_id="include", session_id=first.session_id, message="Мрачный снова можно"))
    constraints = agent.sessions[first.session_id].constraint_state.constraints
    assert second.query.tone is None
    assert not any(c.field == "tone" and c.op == "neq" for c in constraints)


def test_v2_stages_verified_fields_then_commits_only_after_targeted_recovery():
    class RecoveryInterpreter:
        def __init__(self):
            self.calls = 0

        def interpret(self, message, previous, **context):
            del previous
            self.calls += 1
            if self.calls == 1:
                assert not context["unresolved"]
                return StructuredRequest(
                    updates=[
                        ConstraintUpdate(field="kind", value="course", source_text="курс"),
                        ConstraintUpdate(field="genre", value="python", source_text="Python"),
                        ConstraintUpdate(field="level", value="продвинутый", source_text="middle"),
                    ]
                ), 1
            assert context["unresolved"]
            return StructuredRequest(
                reset_constraints=True,
                updates=[ConstraintUpdate(field="level", value="начальный", source_text="начальный")],
            ), 1

    agent = WorkflowAgent(mode="ollama", interpreter=RecoveryInterpreter())
    first = agent.chat(ChatRequest(user_id="recovery", message="Нужен курс Python уровня middle"))
    staged = agent.sessions[first.session_id].constraint_state
    assert first.state == "clarify" and staged.constraints == ()
    assert staged.pending is not None and staged.pending.target_field == "level"
    second = agent.chat(ChatRequest(user_id="recovery", session_id=first.session_id, message="начальный"))
    active = agent.sessions[first.session_id].constraint_state.constraints
    assert second.state == "recommend"
    assert {(c.field, c.value) for c in active} >= {("kind", "course"), ("genre", "python"), ("level", "начальный")}
    assert all(c.source_spans for c in active)


def test_v2_sends_canonical_pending_context_and_keeps_untouched_staging():
    class Backend:
        def __init__(self):
            self.payloads = []

        def structured(self, schema, system, payload):
            del system
            self.payloads.append(payload)
            assert "excluded_genres" not in str(schema.model_json_schema())
            assert "excluded_genres" not in str(payload)
            if len(self.payloads) == 1:
                return schema(
                    updates=[
                        {"field": "kind", "value": "course", "source_text": "курс"},
                        {"field": "genre", "value": "python", "source_text": "Python"},
                        {"field": "level", "value": "продвинутый", "source_text": "middle"},
                    ]
                ), 1
            assert payload["pending_context"]["target_field"] == "level"
            assert {(item["field"], item["value"]) for item in payload["pending_context"]["trusted_staged"]} >= {
                ("kind", "course"),
                ("genre", "python"),
            }
            return schema(updates=[{"field": "level", "value": "начальный", "source_text": "начальный"}]), 1

    backend = Backend()
    agent = WorkflowAgent(mode="ollama", llm=backend)
    first = agent.chat(ChatRequest(user_id="canonical-context", message="Нужен курс Python для middle"))
    second = agent.chat(ChatRequest(user_id="canonical-context", session_id=first.session_id, message="начальный"))
    assert first.state == "clarify" and second.state == "recommend"
    assert {(constraint.field, constraint.value) for constraint in agent.sessions[first.session_id].constraint_state.constraints} >= {
        ("kind", "course"),
        ("genre", "python"),
        ("level", "начальный"),
    }


def test_v2_keeps_second_blocker_and_rejects_an_unbound_yes():
    class TwoBlockers:
        def __init__(self):
            self.calls = 0

        def interpret(self, message, previous, **context):
            del previous
            self.calls += 1
            if self.calls == 1:
                return StructuredRequest(
                    updates=[
                        ConstraintUpdate(field="kind", value="course", source_text="курс"),
                        ConstraintUpdate(field="genre", value="python", source_text="Python"),
                        ConstraintUpdate(field="level", value="продвинутый", source_text="middle"),
                        ConstraintUpdate(field="practical", value=True, source_text="middle"),
                    ]
                ), 1
            if self.calls == 2:
                return StructuredRequest(updates=[]), 1
            if self.calls == 3:
                return StructuredRequest(updates=[ConstraintUpdate(field="level", value="начальный", source_text="начальный")]), 1
            return StructuredRequest(updates=[ConstraintUpdate(field="practical", value=False, source_text="без практики")]), 1

    agent = WorkflowAgent(mode="ollama", interpreter=TwoBlockers())
    first = agent.chat(ChatRequest(user_id="two", message="Нужен курс Python для middle"))
    unbound_yes = agent.chat(ChatRequest(user_id="two", session_id=first.session_id, message="да"))
    assert unbound_yes.state == "clarify"
    level_only = agent.chat(ChatRequest(user_id="two", session_id=first.session_id, message="начальный"))
    assert level_only.state == "clarify"
    pending = agent.sessions[first.session_id].constraint_state.pending
    assert pending is not None and any(f.field == "practical" for f in pending.findings)
    final = agent.chat(ChatRequest(user_id="two", session_id=first.session_id, message="без практики"))
    assert final.state == "recommend" and final.query.practical is False


def test_v2_replaces_numeric_bound_when_user_relaxes_it():
    class NumericInterpreter:
        def interpret(self, message, previous, **context):
            del previous, context
            value = 30 if "30" in message else 60
            updates = [ConstraintUpdate(field="max_minutes", value=value, source_text=f"{value} минут")]
            # The controlled interpreter must cite only the current message;
            # preserving kind is the reducer's responsibility on the next turn.
            if value == 30:
                updates.insert(0, ConstraintUpdate(field="kind", value="film", source_text="фильм"))
            return StructuredRequest(updates=updates), 1

    provider = DemoProvider(
        items=[
            Item(id="short", title="Короткий", kind="film", minutes=25, quality=0.5, description="Фильм"),
            Item(id="long", title="Длинный", kind="film", minutes=50, quality=0.5, description="Фильм"),
        ]
    )
    agent = WorkflowAgent(mode="ollama", interpreter=NumericInterpreter(), provider=provider, question_policy="none")
    first = agent.chat(ChatRequest(user_id="numeric", message="Нужен фильм до 30 минут"))
    second = agent.chat(ChatRequest(user_id="numeric", session_id=first.session_id, message="Теперь можно до 60 минут"))
    assert [item.item.id for item in first.recommendations] == ["short"]
    assert {item.item.id for item in second.recommendations} == {"short", "long"}
    limits = [c.value for c in agent.sessions[first.session_id].constraint_state.constraints if c.field in {"minutes", "max_minutes"}]
    assert limits == [60]
