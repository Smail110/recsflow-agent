"""Build a self-contained HTML report from explicitly selected public receipts.

Example: python -m scripts.build_final_report --source dev=artifacts/run/after.json
Without --source, reuse report/selection.json. No historical/default experiment is
read implicitly, and closed/BLIND/holdout files cannot be selected.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OMITTED = {"dialogues", "records", "rows", "source_files", "structured_calls", "parse_calls", "raw_outputs"}
TITLES = {
    "evaluation-final-20260927": "Проверка версии для сдачи",
    "nli-final": "Последний эксперимент NLI",
    "current-video": "Демонстрация интерфейса",
    "compute-cost": "Условная стоимость вычислений",
    "resource-sample": "Снимок ресурсов во время нагрузки",
    "evaluation-summary": "Сводка DEV и CONTRACT",
    "evaluation-dev": "Ответы агента: DEV",
    "evaluation-contract": "Ответы агента: CONTRACT",
    "evaluation-fallback": "HTTP-проверка отказа LLM в отдельном процессе",
    "evaluation-delivery": "Проверка отдельной копии поставки",
    "evaluation-roles": "LLM-симулятор, валидатор и судья",
    "evaluation-load-c1": "HTTP-нагрузка: concurrency 1",
    "evaluation-load-c2": "HTTP-нагрузка: concurrency 2",
    "evaluation-load-c4": "HTTP-нагрузка: concurrency 4",
}


def table(headers, rows):
    heading = "".join(f"<th>{html.escape(str(cell))}</th>" for cell in headers)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row) + "</tr>" for row in rows)
    return f'<div class="scroll"><table><thead><tr>{heading}</tr></thead><tbody>{body}</tbody></table></div>'


def checked_source(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Selected evidence must be inside the project")
    if any(re.search(r"blind|holdout|secret|credential|private", part, re.I) for part in path.relative_to(root).parts):
        raise ValueError("Closed or private evidence cannot be used in the development report")
    if path.suffix != ".json":
        raise ValueError("Evidence must be a JSON receipt")
    return path


def overview(data: dict) -> dict:
    """Keep reported summaries verbatim; never derive new scores from raw cases."""
    result = {}
    for key, value in data.items():
        if key in OMITTED or (isinstance(value, list) and key not in {"limitations", "commands", "blockers"}):
            continue
        if isinstance(value, dict):
            result[key] = overview(value)
        else:
            result[key] = value
    return result


def scalar_rows(data: dict, prefix=""):
    rows = []
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            rows.extend(scalar_rows(value, name))
        elif not isinstance(value, list):
            rows.append([name, "не измерено / null" if value is None else value])
    return rows


def number(value, decimals=0):
    if value is None:
        return "не измерено / null"
    return f"{value:,.{decimals}f}".replace(",", " ").replace(".", ",")


def source_link(label: str) -> str:
    title = html.escape(TITLES.get(label, label))
    return f'<a href="#receipt-{label}">{title}: JSON и SHA-256</a>'


def findings(receipts: dict[str, dict]) -> str:
    """Present selected measurements without rerunning or rescoring cases."""
    sections = []
    for label, data in receipts.items():
        if data.get("kind") != "submission_snapshot":
            continue
        quality = data["quality"]
        checks = data["engineering"]
        sections.append(
            '<section id="quality"><h2>Качество версии для сдачи</h2>'
            + f"<p>Проверка от {html.escape(data['date'])}. Основная модель — qwen3:8b. "
            "Успех полного диалога требует подходящей выдачи и выполнения критериев сценария; "
            "ответ HTTP 200 сам по себе успехом не считается.</p>"
            + table(
                ["Набор", "Успешные диалоги", "Что проверяет"],
                [
                    ["DEV", f"{quality['dev']['passed']}/{quality['dev']['total']}", "Подбор и уточнение предпочтений в полном диалоге"],
                    ["CONTRACT", f"{quality['contract']['passed']}/{quality['contract']['total']}", "Изменение условий, запреты, продолжение и наблюдаемая выдача"],
                ],
            )
            + "<p>Оба набора небольшие, синтетические и повторно использованы при разработке. "
            "Результат описывает эти сценарии, а не вероятность успеха на реальном пользовательском трафике.</p>"
            + table(
                ["Инженерная проверка", "Результат", "Что подтверждает"],
                [
                    ["Автоматические тесты", f"{checks['tests_passed']} passed; {checks['subtests_passed']} subtests", "Проверенные контракты, состояния и обработку ошибок"],
                    ["Ruff", checks['ruff'], "Соответствие исходников заданным статическим проверкам"],
                    ["git diff --check", checks['diff_check'], "Отсутствие ошибок пробелов и конфликтных маркеров в diff"],
                ],
            )
            + "<p>Прохождение инженерных тестов не означает, что модель всегда верно понимает свободную речь. "
            "Для этого отдельно считаются успех диалога, лишние уточнения и ошибки grounding.</p>"
            + f"<p>Область проверки: {html.escape(checks.get('scope', 'указана в исходной записи'))} "
            + f"Пропущено тестов: {checks.get('tests_skipped', 'см. исходную запись')}.</p>"
            + source_link(label) + "</section>"
        )
    for label, data in receipts.items():
        if not {"DEV", "CONTRACT", "contract_grounding"} <= data.keys():
            continue
        rows = []
        for key in ("DEV", "CONTRACT"):
            result = data[key]
            rows.append(
                [
                    key,
                    f"{result['before']}/{result['total']} → {result['after']}/{result['total']}",
                    len(result["gained"]),
                    len(result["lost"]),
                ]
            )
        grounding = data["contract_grounding"]
        accounting = data["contract_accounting"]
        sections.append(
            '<section id="quality"><h2>Качество полного диалога</h2>'
            "<p>DEV проверяет завершение диалога, CONTRACT — операции над условиями и наблюдаемый ответ. "
            "Успех требует подходящей выдачи, соблюдения явно сказанных условий и ожидаемого состояния. "
            "Ненужное уточнение считается неуспехом, даже если HTTP-запрос обработан корректно.</p>"
            + table(["Набор", "До → после", "Получено успехов", "Потеряно успехов"], rows)
            + "<p>Таблица показывает выбранные сохранённые результаты. Данные и критерии сравнения "
            "указаны в исходной записи; это результат на открытых "
            "контрольных примерах, не оценка улучшения на реальном трафике.</p>"
            + table(
                ["Дополнительная проверка CONTRACT", "Наблюдение"],
                [
                    ["Неподтверждённые факты объяснений", f"{grounding['unsupported']}/{grounding['claims']}"],
                    ["Среднее число уточнений на диалог", number(accounting["mean_clarifications_per_dialogue"], 3)],
                    ["Учтённые LLM-вызовы", number(accounting["llm_calls"].get("total"))],
                    ["Учтённые токены", number(accounting["llm_tokens"].get("total"))],
                ],
            )
            + "<p>Grounding относится к поддерживаемым шаблонам объяснений. Он не доказывает безопасность "
            "произвольного генерируемого текста. Малые повторно используемые наборы не являются независимым "
            "финальным тестом; приведённые в JSON интервалы описательные.</p>" + source_link(label) + "</section>"
        )
    for label, data in receipts.items():
        result = data.get("result", {})
        if "protocol_complete_count" not in result:
            continue
        metrics = result["metrics"]
        sections.append(
            '<section id="roles"><h2>LLM-симулятор и судья</h2>'
            + f"<p>Дата выбранного прогона: {html.escape(data.get('measurement_date', 'см. протокол измерения'))}.</p>"
            + "<p>Симулятор формулирует ответ на реальный уточняющий вопрос из скрытых предпочтений сценария. "
            "Отдельная роль проверяет сохранение смысла ответа; судья оценивает видимый диалог и факты. "
            "Окончательный успех задаёт независимый детерминированный оракул. Судья даёт дополнительную оценку "
            "и не может переопределить его результат.</p>"
            + table(
                ["Показатель", "Результат"],
                [
                    ["Технически завершённые протоколы", f"{result['protocol_complete_count']}/{result['denominator']}"],
                    ["Строго успешные эпизоды", f"{result['success_count']}/{result['denominator']}"],
                    ["Среднее число уточнений", number(metrics.get("mean_clarifications"), 3)],
                    ["Неподтверждённые факты поддерживаемых шаблонов", f"{metrics['unsupported_text_claims']}/{metrics['text_claims']}"],
                    ["Завершённые оценки судьи", number(metrics.get("judge_completed_cases"))],
                ],
            )
            + "<p>Завершение ролей и строгий успех — разные показатели. "
            "Это отдельная малая синтетическая когорта; её нельзя складывать с DEV или CONTRACT. "
            "Агент и роли используют qwen3:8b, поэтому их ошибки могут быть связаны. "
            "Ответы, параметры и модельные identities сохранены для проверки происхождения и replay.</p>"
            + source_link(label)
            + "</section>"
        )
    loads = [(label, data) for label, data in receipts.items() if {"concurrency", "p50_ms", "agent_tokens"} <= data.keys()]
    if loads:
        rows = []
        for _label, data in sorted(loads, key=lambda pair: pair[1]["concurrency"]):
            rows.append(
                [
                    data["concurrency"],
                    data["requests"],
                    f"{data['http_successes']}/{data['requests']}",
                    number(data["p50_ms"] / 1000, 2),
                    number(data["p95_ms"] / 1000, 2),
                    number(data.get("agent_llm_calls")),
                    number(data.get("agent_tokens")),
                ]
            )
        sections.append(
            '<section id="performance"><h2>Задержка, нагрузка и стоимость</h2>'
            + "<p>Даты выбранных прогонов: "
            + html.escape(", ".join(sorted({str(data.get('timestamp_utc', 'не указана'))[:10] for _, data in loads})))
            + ".</p>"
            + "<p>HTTP-нагрузка использует один повторяемый синтетический запрос. "
            "Во всех выбранных опытах работал Ollama; HTTP-успех и непустая выдача "
            "здесь не означают успех произвольного диалога.</p>"
            + table(["Параллельность", "Запросы", "HTTP-успех", "p50, с", "p95, с", "LLM-вызовы", "Токены"], rows)
            + "<p>С ростом параллельности увеличивается время ожидания. Эти измерения относятся к одному "
            "локальному окружению и не задают production SLA. Покрытие токенов и времени инференса отражено "
            "в каждом JSON; агрегаты API не подтверждают полноту сведений об отдельных LLM-вызовах.</p>"
            "<p><strong>Фактическая денежная стоимость неизвестна.</strong> Нет принятой цены инфраструктуры, "
            "измерения электроэнергии и методики распределения расходов. Токены не являются рублями, "
            "а сумма времени инференса не равна оплачиваемому времени работы всей машины. "
            "Снимок ресурсов приведён в исходных записях отдельно; он не измеряет пики нагрузки или полное энергопотребление.</p>"
            + "<p>"
            + " · ".join(source_link(label) for label, _ in loads)
            + "</p></section>"
        )
    for label, data in receipts.items():
        if data.get("kind") != "conditional_compute_cost":
            continue
        sections.append(
            '<section id="cost"><h2>Расчёт стоимости при заданной ставке</h2>'
            '<p>Это сценарный расчёт, не цена оборудования и не фактический счёт. '
            'Я предполагаю выделение одной машины на всё окно нагрузочного прогона. '
            'Стоимость окна = ставка × длительность / 3600; стоимость запроса = стоимость окна / число запросов.</p>'
            + table(
                ["Параллельность", "Ставка, ₽/час", "Окно, ₽", "Запрос, ₽"],
                [[r["concurrency"], r["assumed_hourly_rub"], number(r["window_cost_rub"], 3),
                  number(r["per_request_rub"], 4)] for r in data["scenarios"]],
            )
            + '<p>Ставки в таблице заданы для анализа чувствительности, не взяты из тарифа. '
            'Периоды простоя, загрузка модели до измерения, хранение и сопровождение не включены. '
            'Время параллельных запросов не суммируется как время аренды одной машины.</p>'
            + source_link(label) + '</section>'
        )
    for label, data in receipts.items():
        if data.get("decision") != "REJECT_FOR_PRODUCT_STOP_EXPERIMENTS":
            continue
        synthetic, external = data["synthetic"], data["external"]
        sections.append(
            '<section id="nli"><h2>Эксперимент с NLI</h2>'
            '<p>Я проверил дополнительную модель, которая должна подтверждать извлечённое условие '
            'по цитате пользователя. Последний вариант использовал mDeBERTa, LoRA и прямое '
            'выделение представления цитаты и гипотезы. В рабочем агенте NLI выключен.</p>'
            + table(
                ["Показатель", "До", "После"],
                [
                    ["DEV: неверная цитата подтверждена",
                     f"{synthetic['baseline']['wrong_citation_false_support']}/2484",
                     f"{synthetic['adapted']['wrong_citation_false_support']}/2484"],
                    ["MASSIVE silver: верная цитата подтверждена",
                     f"{external['baseline']['counts']['primary_support']}/985",
                     f"{external['adapted']['counts']['primary_support']}/985"],
                    ["MASSIVE silver: неверная цитата подтверждена",
                     f"{external['baseline']['counts']['wrong_citation_support']}/985",
                     f"{external['adapted']['counts']['wrong_citation_support']}/985"],
                ],
            )
            + '<p>Уменьшение ошибок на внешних неверных цитатах сопровождалось потерей '
            'верных подтверждений. Условия приёмки не выполнены; кандидат отклонён. '
            'Оба набора открытые и уже использовались для анализа. Это результат конкретного '
            'обучения, а не доказательство того, что NLI в целом уступает правилам. '
            'Для продолжения нужны более широкие данные и независимая проверка разметки.</p>'
            + source_link(label) + '</section>'
        )
    return "".join(sections)


def build_report(root: Path, selections: list[dict], output: Path) -> dict:
    root, output = root.resolve(), output.resolve()
    if not output.is_relative_to(root):
        raise ValueError("Report output must be inside the project")
    if not selections:
        raise ValueError("Select at least one actual measurement receipt with --source LABEL=PATH")
    labels = [row["label"] for row in selections]
    if len(set(labels)) != len(labels) or any(not re.fullmatch(r"[a-zA-Z0-9_-]+", label) for label in labels):
        raise ValueError("Evidence labels must be unique ASCII letters, digits, underscores or hyphens")
    evidence = output.parent / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    sources, sections, receipts = [], [], {}
    for selected in selections:
        path = checked_source(root, selected["path"])
        payload = path.read_bytes()
        data = json.loads(payload.decode("utf-8-sig"))
        if not isinstance(data, dict):
            raise ValueError(f"Evidence must contain an object: {path}")
        digest = hashlib.sha256(payload).hexdigest()
        if selected.get("expected_sha256", digest) != digest:
            raise ValueError(f"Selected receipt changed: {path}; select explicitly to accept new evidence")
        label = selected["label"]
        receipts[label] = data
        bundled = evidence / f"{label}.json"
        bundled.write_bytes(payload)
        summary = overview(data)
        media = ""
        if "duration_seconds" in data and path.name == "video.json":
            video = path.with_name("demo.mp4")
            if video.is_file() and video.parent == output.parent:
                if data.get("bytes") is not None and data["bytes"] != video.stat().st_size:
                    raise ValueError("Selected video receipt does not match demo.mp4 size")
                media = '<p><a href="demo.mp4">Открыть запись реального интерфейса</a></p>'
        metric_sections = {
            k: v
            for k, v in summary.items()
            if k
            in {
                "summary",
                "metrics",
                "comparison",
                "checks",
                "acceptance",
                "baseline",
                "after",
                "resource_telemetry",
                "usage_coverage",
                "cost_assumptions",
                "results",
            }
            or not isinstance(v, (dict, list))
        }
        sections.append(
            f'<details id="receipt-{label}" class="receipt"><summary>{html.escape(TITLES.get(label, label))}</summary>'
            f"<p>Источник: <code>{html.escape(path.relative_to(root).as_posix())}</code><br>"
            f"SHA-256: <code>{digest}</code></p>"
            f"{table(['Показатель из receipt', 'Значение'], scalar_rows(metric_sections))}"
            f"<details><summary>Контекст, ограничения и происхождение</summary><pre>"
            f"{html.escape(json.dumps(summary, ensure_ascii=False, indent=2))}</pre></details>"
            f'<p><a href="evidence/{label}.json">Полная запись измерения (JSON)</a></p>{media}</details>'
        )
        sources.append(
            {
                "label": label,
                "path": path.relative_to(root).as_posix(),
                "sha256": digest,
                "bundled_path": bundled.relative_to(root).as_posix(),
            }
        )
    generated = datetime.now(UTC).isoformat()
    style = """body{max-width:1080px;margin:auto;padding:30px;color:#172f36;background:#f7f7f3;font:16px/1.6 system-ui}h1{font-size:52px;line-height:1.1}h2{margin-top:38px}header{border-bottom:3px solid #168b70;padding-bottom:24px}table{border-collapse:collapse;width:100%;background:white}th,td{padding:11px;text-align:left;border-bottom:1px solid #ddd;overflow-wrap:anywhere}th{background:#e7efeb}.scroll{overflow:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;padding:16px;background:#e7efeb}code{overflow-wrap:anywhere}a{color:#087264}section{margin:36px 0}aside{padding:16px;background:#fff0cc}.receipt{background:white;padding:14px 18px;margin:12px 0;border:1px solid #d8e4df}.receipt>summary{cursor:pointer;font-weight:650}.flow{padding:20px;background:#e7efeb;border-left:4px solid #168b70}nav{display:flex;gap:18px;flex-wrap:wrap;padding:20px 0}footer{font-size:13px;color:#4b646c;border-top:1px solid #d8e4df;padding-top:20px}@media(max-width:650px){body{padding:20px}h1{font-size:38px}th,td{min-width:75px;font-size:14px}}@media print{body{background:white;font-size:10pt}nav{display:none}h2{break-after:avoid}details{break-inside:avoid}}"""
    body = f"""<header><p>Диалоговый рекомендательный агент · прототип для демонстрации</p><h1>RecAgent</h1>
<p>Автор: Герман Роев Александрович.</p>
<p>Один прототип: от естественного запроса до рекомендаций с проверяемыми объяснениями.</p></header>
<nav aria-label="Разделы отчёта"><a href="#task">Задача</a><a href="#quality">Качество</a><a href="#roles">LLM-роли</a>
<a href="#performance">Нагрузка</a><a href="#next">Ограничения и план</a><a href="#evidence">Источники</a></nav>
<section id="task"><h2>Задача и архитектура</h2>
<p>Пользователь описывает цель, уточняет предпочтения и меняет условия в диалоге. Агент должен находить
подходящие фильмы, сериалы или курсы, объяснять выбор фактами каталога и сохранять условия при продолжении поиска.</p>
<p>Чат и REST API управляют LangGraph workflow-v2: структурированное понимание запроса,
память условий, кандидаты и метаданные, строгие фильтры, ранжирование, уточнения и объяснения с фактами каталога.
Основная локальная модель — qwen3:8b; rules — отдельный режим деградации.</p>
<p class="flow">Streamlit / REST → LangGraph → LLM и проверка структуры → память условий →
provider / каталог → строгая фильтрация и RRF → уточнение или рекомендация с фактами</p>
<p>Recsflow представлен заменяемым адаптером и проверяемым HTTP mock. Реальный endpoint, credentials и контракт
не предоставлены; совместимость с закрытым production API не заявляется.</p>
<p>Научные предпосылки, реперные работы и выбор метрик описаны в <a href="../docs/RESEARCH-BASIS.md">обзоре исследований</a>.</p>
<h3>Данные и методика</h3><p>DEV — 20 синтетических диалогов с каталогом seed 42;
CONTRACT — 12 AI-написанных сценариев, 20 ходов и каталог seed 137. Названия, профили и ряд атрибутов
созданы генератором. MovieLens 100K калибрует отдельные распределения, но не служит каталогом выдачи
или эталоном диалогов. ReDial и CCPE не входят в показанные продуктовые метрики.</p>
<aside>Результаты ниже относятся только к явно выбранным запускам. Инженерные тесты, HTTP 200,
успех диалога и процент готовности — разные величины. Неизвестные ресурсы и стоимость обозначены null.
Повторно используемый DEV не является независимой финальной оценкой. Даты в исходных записях обозначают
происхождение отдельных измерений, а не разные продукты или единую одновременную приёмку.
Закрытые наборы этот отчёт не читает.</aside></section>
{findings(receipts)}
<section id="next"><h2>Узкие места и план продуктивизации</h2>
<p>Часть полных DEV-диалогов и эпизодов с LLM-симулятором ещё не проходит строгий критерий успеха.
Проверки иногда приводят к лишнему уточнению; сложные разговорные формулировки требуют более широкой оценки.
Политика уточнения использует разнообразие кандидатов и стоимость вопроса как эвристику; её преимущество
над фиксированным порядком не доказано.</p>
<ol><li><strong>Довести качество диалога.</strong> Разобрать оставшиеся ошибки по этапам, исправлять общий механизм,
сравнивать на неизменных DEV/CONTRACT и соседних формулировках. Принимать улучшение без потери прежних успехов
и новых нарушений жёстких условий или grounding. Малую повторно используемую синтетику дополнить отдельной
русской выборкой с независимой разметкой и разбором разногласий.</li>
<li><strong>Проверить стоимость второго LLM-вызова.</strong> Сравнить выбор готовых evidence через LLM
и детерминированный вариант на одинаковых входах. Это гипотеза следующего опыта: ускорение принимается
только при сохранении качества полного диалога. Замена default-модели заранее не предполагается.</li>
<li><strong>Развить NLI-проверку.</strong> Собрать более широкий набор запросов с разметкой роли, цитаты,
отрицания и операции над условием. Независимо проверить разметку, разделить данные по сценариям и формулировкам,
затем обучать и оценивать модель. Улучшение на знакомой синтетике само по себе недостаточно для подключения.</li>
<li><strong>Подключить настоящую платформу.</strong> Нужны staging endpoint, схема авторизации,
описание retrieval/lookup/history/title search, семантика полей, тестовый каталог и ограничения нагрузки.
После проверки адаптера — авторизация пользователя, постоянные сессии, мониторинг, лимиты и проверка SLO.
Предложенный OpenAPI описывает mock-границу, а не подтверждённый Recsflow API.</li></ol>
<p>Для демонстрации доступны UI, API, контролируемая деградация и замена provider. Production readiness
и качество на реальных пользователях этими измерениями не установлены. Процент готовности не выводится.</p></section>
<section id="evidence"><h2>Проверяемые источники</h2><p>Сводка выше читает только явно выбранные JSON.
Каждая запись ниже доступна рядом с HTML; selection.json фиксирует её SHA-256. Наблюдения сохранены
без повторного inference или пересчёта критериев по ответам агента.</p>{"".join(sections)}</section>
<section><h2>Воспроизведение</h2><pre>python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
ollama pull qwen3:8b
docker compose up --build -d
# UI http://localhost:8501; API http://localhost:8000/docs
python -m scripts.run_demo_notebook --mode ollama --output report/demo-ollama-local.ipynb
python -m scripts.build_final_report
python -m scripts.prepare_git_snapshot --output dist/git-candidate-YYYYMMDD</pre>
<p>README и runbook описывают режимы, проверки, ограничения и точные команды экспериментов.
Встроенные JSON содержат измеренные значения и параметры; выбор источников — selection.json.
Синтетический каталог воспроизводится генератором seed 42. Зависимости Jupyter нужны только для интерактивного
редактирования notebook; последовательный запуск его Python-ячеек выполняет scripts.run_demo_notebook.</p></section>
<footer>Отчёт собран {html.escape(generated)} из сохранённых измерений. Это время генерации документа,
не дата повторной проверки агента.</footer>"""
    output.write_text(
        f'<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RecAgent — проверяемые результаты</title><style>{style}</style></head><body>{body}</body></html>',
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "generated_at_utc": generated,
        "sources": sources,
        "report_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    (output.parent / "selection.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--selection", type=Path, default=ROOT / "report" / "selection.json")
    parser.add_argument("--output", type=Path, default=ROOT / "report" / "index.html")
    args = parser.parse_args()
    if args.source:
        selections = []
        for row in args.source:
            label, separator, path = row.partition("=")
            if not separator:
                parser.error("--source must be LABEL=PATH")
            selections.append({"label": label, "path": path})
    elif args.selection.exists():
        previous = json.loads(args.selection.read_text(encoding="utf-8"))
        selections = [
            {
                "label": row["label"],
                "expected_sha256": row["sha256"],
                "path": row["path"] if (ROOT / row["path"]).exists() else row["bundled_path"],
            }
            for row in previous["sources"]
        ]
    else:
        parser.error("No selected evidence. Pass --source LABEL=PATH; no old report is read automatically.")
    result = build_report(ROOT, selections, args.output)
    print(f"Отчёт: {args.output}; проверяемых источников: {len(result['sources'])}")


if __name__ == "__main__":
    main()
