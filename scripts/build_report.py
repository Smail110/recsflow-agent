"""Build a standalone HTML report directly from recorded experiment files."""

import html
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def load(name):
    path = Path("report") / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def format_value(value):
    if value is None:
        return "не измерено"
    if isinstance(value, float):
        return f"{value:.3f}"
    return html.escape(str(value))


def main():
    from recagent.catalog import CATALOG_SIZE

    # Размер каталога подставляется, а не пишется строкой: каталог перегенерируется,
    # и зашитое «84» уже однажды стало неправдой при 3000 объектах.
    catalog_size = CATALOG_SIZE
    rules, llm, load_test = load("evaluation.json"), load("evaluation-ollama.json"), load("load-test.json")
    suites = ET.parse("report/tests.xml").getroot().findall("testsuite") if Path("report/tests.xml").exists() else []
    tests = sum(int(s.attrib["tests"]) for s in suites)
    failures = sum(int(s.attrib["failures"]) + int(s.attrib["errors"]) for s in suites)
    rows = []
    for label, field in [
        ("Сценарии", "scenarios"),
        ("Success rate, доля", "success_rate"),
        ("Уточнений на сценарий", "mean_clarifications"),
        ("Проверенных утверждений", "claim_count"),
        ("Доля неподтверждённых утверждений", "unsupported_claim_rate"),
        ("p50 обработки, мс", "latency_p50_ms"),
        ("p95 обработки, мс", "latency_p95_ms"),
        ("Попытки LLM агента", "agent_llm_calls"),
        ("Токены агента", "agent_tokens"),
        ("Ходы с fallback", "fallback_turns"),
        ("Завершённые оценки LLM-судьи", "llm_judge_completed"),
        ("Success rate LLM-судьи", "llm_judge_success_rate"),
        ("Токены симулятора и судьи", "evaluator_tokens"),
        ("Денежная стоимость", "monetary_cost"),
    ]:
        rows.append(
            f"<tr><td>{label}</td><td>{format_value(rules['metrics'].get(field)) if rules else 'нет прогона'}</td><td>{format_value(llm['metrics'].get(field)) if llm else 'нет прогона'}</td></tr>"
        )
    scenario_rows = []
    if llm:
        for record in llm["records"]:
            judge = record.get("llm_judge")
            scenario_rows.append(
                f"<tr><td>{html.escape(record['id'])}</td><td>{'Пройден' if record['success'] else 'Не пройден'}</td><td>{'Успех' if judge and judge['success'] else 'Неуспех' if judge else 'Нет оценки'}</td><td>{html.escape(record['response']['mode'])}</td></tr>"
            )
    load_rows = "".join(
        f"<tr><td>{label}</td><td>{format_value(load_test.get(key)) if load_test else 'не измерено'}</td></tr>"
        for label, key in [
            ("Запросы", "requests"),
            ("Одновременные клиенты", "concurrency"),
            ("Успешные ответы", "successes"),
            ("Ошибки", "errors"),
            ("p50, мс", "p50_ms"),
            ("p95, мс", "p95_ms"),
            ("Запросов/сек", "requests_per_second"),
        ]
    )
    report = f"""<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RecAgent — отчёт о прототипе</title>
<style>body{{font:16px/1.65 system-ui,sans-serif;color:#20232b;background:#f6f7f9;margin:0}}main{{max-width:980px;margin:36px auto;background:white;padding:48px;border-radius:20px}}h1{{font-size:44px;line-height:1.1;letter-spacing:-2px}}h2{{margin-top:38px;font-size:24px}}.label{{color:#ed174b;font-size:12px;font-weight:750;letter-spacing:2px}}.note{{padding:18px 22px;background:#fff2f5;border-left:3px solid #ed174b}}table{{width:100%;border-collapse:collapse;font-size:14px;margin:18px 0}}th,td{{padding:10px 13px;border-bottom:1px solid #e8e9ed;text-align:left}}th{{background:#f6f7f9}}a{{color:#c5103b}}code{{background:#f3f4f6;padding:2px 5px}}.flow{{padding:22px;background:#f3f4f6;border-radius:12px;line-height:2}}small{{color:#646978}}@media(max-width:700px){{main{{margin:0;padding:22px;border-radius:0}}h1{{font-size:34px}}table{{font-size:12px}}th,td{{padding:7px}}}}@media print{{body{{background:white}}main{{margin:0;padding:0}}h2{{break-after:avoid}}tr{{break-inside:avoid}}}}</style>
<main><div class="label">RECAGENT / ПРОТОТИП / 11 СЕНТЯБРЯ 2026</div><h1>Рекомендации<br>через диалог</h1>
<p>Рабочее автономное ядро для последующего подключения к Recsflow. Чат на Streamlit, REST API на FastAPI, граф на LangGraph, локальная Qwen через Ollama.</p>
<p class="note"><b>Исходные данные:</b> предоставлено только описание задания. Каталог, история и контракт Recsflow отсутствуют. Использованы {catalog_size} вымышленных объекта (генератор, seed=42, распределения откалиброваны по MovieLens-100k) и синтетическая история демо-профиля. Этот отчёт не подтверждает качество платформы Recsflow.</p>
<h2>1. Задача и результат</h2><p>Прототип принимает пожелания на русском, сохраняет ограничения в сессии, уточняет недостаточный запрос, получает и ранжирует кандидатов, возвращает до пяти вариантов с объяснениями. Поддерживаются фильмы, сериалы, курсы, похожие объекты, реакции и исключение знакомого контента.</p>
<div class="flow">Запрос → structured output / правила → валидация → уточнение или retrieval → metadata → строгие фильтры → реранжирование → проверка фактов → ответ → обратная связь</div>
<p>Слой <code>RecommendationProvider</code> отделяет ядро от источника данных. Его интерфейс — предложение для интеграции, не реконструкция недоступного API. Реальный адаптер должен реализовать retrieval, lookup, history и find_title.</p>
<h2>2. Что реализовано</h2><table><tr><th>Компонент</th><th>Реализация</th></tr>
<tr><td>Извлечение запроса</td><td>Pydantic, Ollama JSON Schema, явные ограничения поверх результата модели, резервные правила</td></tr>
<tr><td>Инструменты</td><td>Синтетический retrieval, metadata lookup, строгие фильтры, эвристический реранкер, уточнение</td></tr>
<tr><td>Диалог</td><td>LangGraph, память одного процесса, блокировки, TTL, реакции, лимит 8 LLM-попыток</td></tr>
<tr><td>Объяснения</td><td>Проверяемые шаблоны; evidence для каждого факта и связи с историей/seed</td></tr>
<tr><td>Оценка</td><td>Фиксированные скрытые предпочтения, LLM-перефразирование первого хода, LLM-судья, отдельный oracle</td></tr>
<tr><td>Передача</td><td>README, Dockerfile/Compose, notebook, тесты, сценарий демонстрации, контракт адаптера</td></tr></table>
<h2>3. Методика оценки</h2><p>12 фиксированных сценариев проверяют ограничения, уточнения, смену формата, память, пустой результат, снятие ограничения, исключения и поиск похожего. Успех означает непустую выдачу, где каждый объект соответствует скрытым условиям; для заведомо пустого и неизвестного запроса проверяется корректное отсутствие выдачи или уточнение.</p>
<p>Oracle проверяет атрибуты по условиям сценария независимо от фильтра агента. Оценка объяснений — сверка структурированных evidence с каталогом и контекстом. Свободных LLM-объяснений нет: нулевая доля неверных фактов характеризует шаблоны, а не общую способность модели не галлюцинировать.</p>
<p>LLM-симулятор перефразирует первый ход, последующие ходы фиксированы. Судья получает предпочтения и результат отдельным вызовом. Симулятор и судья используют одну модель: возможны коррелированные ошибки и изменение смысла перефразирования.</p>
<table><tr><th>Метрика</th><th>Без LLM</th><th>Ollama + симулятор + судья</th></tr>{"".join(rows)}</table>
<small>Модель: {html.escape(str(llm.get("model"))) if llm else "не проверена"}. Окружение: {html.escape(str(rules.get("environment"))) if rules else "нет записи"}. Задержки здесь измерены внутри ядра на ход, без времени симулятора и судьи. Это повторяемые инженерные сценарии, не независимый holdout.</small>
<table><tr><th>Сценарий LLM</th><th>Oracle</th><th>LLM-судья</th><th>Режим</th></tr>{"".join(scenario_rows)}</table>
<h2>4. Выявленные ошибки и ограничения судьи</h2><p>Первый LLM-прогон выявил неверное отрицание «не слишком мрачный». Затем обнаружились перенос старого тона при смене сериала на курс и чувствительность поиска названия к падежу. Исправлены отрицание и перенос ограничений; добавлен консервативный поиск единственного близкого названия, без генерации метаданных. Добавлены регрессионные тесты. Первоначальные результаты сохранены в отдельных JSON.</p>
<p class="note">LLM-судья объявлял успешными и некоторые ошибочные ответы, включая пустую выдачу при наличии подходящих вариантов. Его положительный вердикт не является критерием приёмки. Основной показатель в отчёте — независимая проверка условий. Исправления делались по этому же набору: итоговая оценка оптимистична и требует нового holdout и оценки людьми.</p>
<h2>5. HTTP-нагрузка и тесты</h2><p>Реальный локальный HTTP-прогон режима rules, один процесс Uvicorn, синтетический каталог. На каждый запрос открывается новое соединение; в задержку включены расходы клиента. Эти результаты не экстраполируются на LLM, удалённую платформу или большой каталог.</p><table><tr><th>Показатель</th><th>Значение</th></tr>{load_rows}</table>
<p>Автоматические тесты: <b>{tests}</b>, ошибок/провалов: <b>{failures}</b>. Проверены ядро, API, состояние сессий, отказ модели/источника, grounding, повторная фильтрация и Streamlit-чат. Интерфейс дополнительно открыт и проверен в браузере.</p>
<p>Токены агента и оценщиков учитываются отдельно. Денежная стоимость не определена: электроэнергия, GPU-время и амортизация не измерялись. Нулевые LLM-вызовы в режиме rules не означают нулевую стоимость инфраструктуры.</p>
<h2>6. Узкие места и план продуктивизации</h2><ol>
<li>Согласовать API Recsflow, схему метаданных и событий. Заменить источник, измерить recall до/после фильтров на реальном каталоге.</li>
<li>Собрать независимый набор разговоров и скрытых предпочтений, добавить людей-оценщиков, отдельно проверить отрицания и противоречия.</li>
<li>Настроить модель/промпт по результатам holdout, оценить латентность и отказоустойчивость на целевом оборудовании.</li>
<li>Заменить память процесса на Redis/БД, добавить авторизацию, квоты пользователя, аудит, наблюдаемость и межпроцессную синхронизацию.</li>
<li>Вывести тон и длину объяснений в конфигурацию бренда, согласовать SLO и провести повторный HTTP-прогон с реальной LLM и платформой.</li></ol>
<p>Не реализованы: подключение к Recsflow, обученное ранжирование, production-auth, персистентная память, полноценный адаптивный симулятор диалога, конфигурация бренда. Контейнеры подготовлены, но не собирались в рамках проверки. Сценарий скринкаста есть; видеозаписи нет.</p>
<h2>7. Воспроизведение и источники</h2><p>Команды и точные зависимости — в <a href="../README.md">README</a>. Исходные результаты: <a href="evaluation.json">rules</a>, <a href="evaluation-ollama.json">Ollama</a>, <a href="load-test.json">HTTP-нагрузка</a>. Генератор данных: <code>python -m recagent.catalog --seed 42</code>. Сборка этого отчёта: <code>python -m scripts.build_report</code>.</p>
<ul><li><a href="https://aclanthology.org/2025.findings-emnlp.620/">Peng et al., A Survey on LLM-powered Agents for Recommender Systems, EMNLP 2025</a> — контекст применения агентов в рекомендациях.</li>
<li><a href="https://arxiv.org/abs/2501.09493">Evaluating Conversational Recommender Systems with Large Language Models, 2025</a> — пользовательская оценка CRS; эксперименты статьи здесь не воспроизводились.</li>
<li><a href="https://docs.ollama.com/capabilities/structured-outputs">Ollama Structured Outputs</a> и <a href="https://docs.langchain.com/oss/python/langgraph/graph-api">LangGraph Graph API</a> — документация применённых механизмов.</li></ul>
</main></html>"""
    Path("report/index.html").write_text(report, encoding="utf-8")
    print("report/index.html")


if __name__ == "__main__":
    main()
