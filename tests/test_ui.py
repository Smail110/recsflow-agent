from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_demo_chat_and_new_session():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run(timeout=20)
    assert not app.exception
    app.chat_input[0].set_value("Хочу лёгкий детективный сериал, один сезон").run(timeout=20)
    assert not app.exception
    assert len(app.chat_message) == 2
    result = app.session_state["messages"][-1]["content"]
    assert result.recommendations and result.query.max_seasons == 1
    app.chat_input[0].set_value("Не дольше 30 минут").run(timeout=20)
    assert not app.exception
    result = app.session_state["messages"][-1]["content"]
    assert result.recommendations and all(r.item.minutes <= 30 for r in result.recommendations)
