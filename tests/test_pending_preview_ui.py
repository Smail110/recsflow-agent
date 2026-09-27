"""The real Streamlit app separates proposed changes from active Query."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from recagent.interpretation import ConstraintUpdate, StructuredRequest
from recagent.models import PendingChangePreview, PendingPreview
from recagent.response_generation import EvidenceResponseGenerator
from recagent.workflow import WorkflowAgent


def test_app_shows_readable_pending_conditions_without_active_query_mutation(monkeypatch):
    class Interpreter:
        def interpret(self, *_args, **_kwargs):
            return StructuredRequest(
                updates=[
                    ConstraintUpdate(field="kind", value="film", source_text="фильм"),
                    ConstraintUpdate(field="genre", operation="exclude", value="комедия", source_text="без комедии"),
                ],
                issues=[{"kind": "ambiguity", "field": "tone", "source_text": "особый", "message": "Какой тон?"}],
            ), 3

    monkeypatch.setenv("RECAGENT_MODE", "rules")
    monkeypatch.setenv("RECAGENT_SHOW_DIAGNOSTICS", "1")
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run(timeout=20)
    app.session_state["agent"] = WorkflowAgent(mode="ollama", interpreter=Interpreter(), response_generator=EvidenceResponseGenerator())
    app.chat_input[0].set_value("Нужен фильм без комедии, особый тон").run(timeout=20)
    assert not app.exception
    result = app.session_state["messages"][-1]["content"]
    assert result.state == "clarify" and result.query.kind is None
    assert result.pending_preview and not result.recommendations
    assert any("ещё не применены" in caption.value for caption in app.caption)
    assert any("фильм" in text.value and "film" not in text.value for text in app.text)
    assert any("Исключить" in text.value and "комедия" in text.value for text in app.text)
    assert any("Действующие условия" in caption.value for caption in app.caption)

    # Render presentation-only operations directly; no fake model inference.
    result.pending_preview = PendingPreview(
        question_id="presentation",
        base_version=0,
        changes=[
            PendingChangePreview(
                field="genre", label="жанр", operation="remove", operator="neq", value="комедия", source_text=["вернуть комедию"]
            ),
            PendingChangePreview(field="max_minutes", label="длительность", operation="clear", operator="lte", source_text=["без лимита"]),
        ],
    )
    app.run(timeout=20)
    assert not app.exception
    assert any(text.value == "Снять исключение: жанр — комедия" for text in app.text)
    assert any(text.value == "Убрать ограничение: длительность" for text in app.text)


def test_app_hides_diagnostics_by_default(monkeypatch):
    monkeypatch.setenv("RECAGENT_MODE", "rules")
    monkeypatch.delenv("RECAGENT_SHOW_DIAGNOSTICS", raising=False)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run(timeout=20)
    assert not app.exception
    assert not any(caption.value == "Действующие условия подбора" for caption in app.caption)
    assert not any(box.label == "Политика уточнений" for box in app.selectbox)
