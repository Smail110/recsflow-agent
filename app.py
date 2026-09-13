import os

import streamlit as st

from recagent.agent import Agent
from recagent.models import ChatRequest

st.set_page_config(page_title="RecAgent · Подбор в диалоге", page_icon="✳", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """<style>
.stApp {background:#f6f7f9;} [data-testid="stSidebar"]{background:#fff;}
.block-container{max-width:1100px;padding-top:2.5rem;padding-bottom:5rem;}
h1{letter-spacing:-.055em;font-weight:750!important;}
[data-testid="stChatMessage"]{background:white;border:1px solid #e8e9ed;border-radius:16px;padding:1.2rem;}
.eyebrow{color:#ed174b;font-size:.75rem;font-weight:750;letter-spacing:.16em;margin-bottom:.9rem;}
.intro{font-size:1.05rem;color:#656975;max-width:640px;line-height:1.65;}
.note{background:#f4f5f8;border-radius:12px;padding:16px;color:#696d78;font-size:.84rem;line-height:1.65;}
</style>""",
    unsafe_allow_html=True,
)


def new_session():
    st.session_state.agent = Agent(mode=st.session_state.get("mode", "rules"))
    st.session_state.sid = None
    st.session_state.messages = []


if "agent" not in st.session_state:
    st.session_state.mode = os.getenv("RECAGENT_MODE", "rules")
    new_session()

with st.sidebar:
    st.markdown("## ✳ RecAgent")
    st.caption("РЕКОМЕНДАЦИИ В ДИАЛОГЕ")
    st.divider()
    st.selectbox(
        "Режим подбора",
        ["rules", "ollama"],
        format_func=lambda x: "Демо · без модели" if x == "rules" else "Локальная LLM · Ollama",
        key="mode",
        on_change=new_session,
    )
    st.selectbox(
        "Профиль",
        ["demo", "new-user"],
        format_func=lambda x: "Демо · есть история" if x == "demo" else "Новый пользователь",
        key="profile",
        on_change=new_session,
    )
    if st.button("＋ Новый диалог", use_container_width=True):
        new_session()
        st.rerun()
    st.divider()
    st.markdown("**Как попробовать**")
    st.markdown("1. Опишите, что хочется.\n2. Добавьте ограничения.\n3. Оцените подборку или попросите ещё.")
    # Число объектов берётся из каталога: каталог перегенерируется, и зашитая
    # цифра однажды станет неправдой (раньше здесь было «84» при 3000 объектах).
    st.markdown(
        f'<div class="note">Демонстрационный стенд<br><b>{len(st.session_state.agent.provider.items)} вымышленных объекта</b><br>Фильмы, сериалы и курсы. История демо-профиля тоже синтетическая.</div>',
        unsafe_allow_html=True,
    )
    with st.expander("Посмотреть каталог"):
        st.dataframe(
            [{"Название": i.title, "Жанр": i.genre, "Формат": i.kind} for i in st.session_state.agent.provider.items.values()],
            hide_index=True,
        )

st.markdown('<div class="eyebrow">RECAGENT / ПРОТОТИП 01</div>', unsafe_allow_html=True)
st.title("Найдём то, что вам близко.")
st.markdown(
    '<p class="intro">Расскажите о настроении, планах или интересах.<br>Я учту пожелания и объясню каждый выбор.</p>',
    unsafe_allow_html=True,
)

suggested = None
if not st.session_state.messages:
    st.write("")
    examples = [
        ("На спокойный вечер", "Хочу лёгкий детективный сериал, не длиннее одного сезона"),
        ("Немного фантастики", "Фильм, фантастика, не дольше 90 минут"),
        ("Освоить новое", "Курс по машинному обучению для новичка, без воды"),
    ]
    for column, (label, prompt) in zip(st.columns(3), examples, strict=True):
        with column, st.container(border=True):
            st.markdown(f"**{label}**")
            st.caption(prompt)
            if st.button("Попробовать →", key=label, use_container_width=True):
                suggested = prompt
    st.caption("Можно начать с простого: «Посоветуй что-нибудь». Я задам уточняющий вопрос.")


def render_response(response, message_index):
    st.write(response.message)
    for warning in response.warnings:
        st.warning(warning)
    for rank, recommendation in enumerate(response.recommendations, 1):
        item = recommendation.item
        with st.container(border=True):
            st.markdown(f"**{rank:02d} · {item.title}**")
            st.caption(
                f"{ {'series': 'Сериал', 'film': 'Фильм', 'course': 'Курс'}[item.kind] } · {item.genre.capitalize()} · Вымышленный объект"
            )
            st.write(recommendation.explanation)
            # Четвёртая узкая колонка — отступ: кнопки не растягиваются на всю ширину
            # карточки. Кнопкам отдаются только первые три, поэтому zip со strict=True.
            reaction_columns = st.columns([1, 1, 1, 2])[:3]
            reactions = [("♡ Нравится", "like"), ("Не подходит", "dislike"), ("Уже знакомо", "seen")]
            for column, (label, reaction) in zip(reaction_columns, reactions, strict=True):
                if column.button(label, key=f"{message_index}-{item.id}-{reaction}"):
                    st.session_state.agent.feedback(response.session_id, item.id, reaction)
                    st.toast("Учту в следующей подборке")
            with st.expander("На чём основан выбор"):
                st.dataframe(
                    [
                        {"Поле": e.field, "Значение": str(e.value), "Связь": e.relation, "Источник": e.source_item_id or e.item_id}
                        for e in recommendation.evidence
                    ],
                    hide_index=True,
                    use_container_width=True,
                )
    with st.expander("Диагностика запроса"):
        st.caption(
            f"{response.mode} · {response.latency_ms:.0f} мс · LLM-вызовов: {response.llm_calls_total}/8 · токенов за ход: {response.llm_tokens}"
        )
        st.json(response.query.model_dump(exclude_none=True))
        st.caption(" → ".join(response.trace))
    st.download_button(
        "Скачать ответ JSON",
        response.model_dump_json(indent=2),
        file_name="recagent-response.json",
        mime="application/json",
        key=f"download-{message_index}",
    )


for index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.write(message["content"])
        else:
            render_response(message["content"], index)

prompt = st.chat_input("Например: хочу что-нибудь лёгкое на вечер", max_chars=2000) or suggested
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    with st.spinner("Подбираю варианты и проверяю факты…"):
        try:
            result = st.session_state.agent.chat(
                ChatRequest(message=prompt, session_id=st.session_state.sid, user_id=st.session_state.get("profile", "demo"))
            )
            st.session_state.sid = result.session_id
            st.session_state.messages.append({"role": "assistant", "content": result})
        except Exception as exc:
            st.error(f"Не удалось обработать запрос: {exc}")
            st.stop()
    st.rerun()
