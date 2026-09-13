"""Запрет циркулярности между оценочным harness и кодом агента.

Почему это отдельный тест, а не соглашение в докстринге. Весь смысл переделки
оценки в том, чтобы критерий успеха нельзя было удовлетворить «по построению»
(``docs/EVAL-PLAN.md`` §1). Единственная реальная гарантия — запрет импортов.
Соглашение в комментарии никто не нарушает специально, его нарушают случайно,
в спешке, и метрики тихо становятся тавтологией. Тест делает нарушение падающим
CI, а не вопросом добросовестности.

Проверяются исходники на уровне текста, а не объектов: импорт ``recagent.agent``
внутри функции или под ``if TYPE_CHECKING`` в рантайме не виден, а в исходнике —
виден. Поэтому тест читает файлы.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EVALS = ROOT / "evals"

#: Модули агента, которые оракулу и симулятору импортировать запрещено.
#: ``recagent.catalog.users`` разрешён: theta — это данные (D2), а не код агента;
#: без неё оракулу неоткуда взять ground truth. ``recagent.models`` разрешён
#: частично: ``Item`` нужен как тип данных каталога, но ``Query`` — нет.
FORBIDDEN_MODULES = frozenset(
    {
        "recagent.agent",
        "recagent.parsing",
        "recagent.grounding",
        "recagent.providers",
        "recagent.api",
    }
)

#: Имена, которые запрещено импортировать из любых модулей recagent.
FORBIDDEN_NAMES = frozenset(
    {
        "Agent",  # сам агент
        "matches",  # жёсткие фильтры агента: оракул обязан иметь свою реализацию
        "rule_parse",  # парсер: оракул не должен знать, что агент умеет извлекать
        "explain",  # генератор объяснений
        "validate_evidence",  # проверка evidence агента
        "Query",  # извлечённый агентом запрос
        "ChatResponse",  # ответ агента целиком
        "DemoProvider",
    }
)


def _imported_names(tree: ast.AST) -> list[tuple[str, int, str]]:
    """Все импорты файла: ``(модуль, строка, имя)``.

    Отдельно обрабатываются ``from X import a, b`` и ``import X`` — во втором
    случае имя совпадает с модулем. ``import X as Y`` даёт имя X, а не Y:
    переименование не должно обходить запрет.
    """
    found: list[tuple[str, int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append((alias.name, node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                # Относительный импорт внутри evals: нас не интересует.
                continue
            module = node.module or ""
            for alias in node.names:
                found.append((module, node.lineno, alias.name))
    return found


def _evals_sources() -> list[Path]:
    return sorted(path for path in EVALS.rglob("*.py") if path.name != Path(__file__).name)


def test_evals_package_exists_and_is_not_empty():
    """Санитарная проверка: если папка переименована, тест ниже молча пройдёт."""
    sources = _evals_sources()
    assert sources, f"в {EVALS} нет ни одного .py файла"
    assert (EVALS / "oracle.py").exists()


@pytest.mark.parametrize("path", _evals_sources(), ids=lambda p: p.name)
def test_evals_do_not_import_agent_code(path: Path):
    """Ни один файл evals не импортирует код агента.

    Тест параметризован по файлам, а не собран в один: падение указывает на
    конкретный файл, и добавление нового модуля в evals автоматически попадает
    под проверку.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    violations: list[str] = []

    for module, lineno, name in _imported_names(tree):
        root_module = module.split(".")[0]
        if module in FORBIDDEN_MODULES or any(module.startswith(prefix + ".") for prefix in FORBIDDEN_MODULES):
            violations.append(f"{lineno}: импорт модуля агента {module!r}")
        elif root_module == "recagent" and name in FORBIDDEN_NAMES:
            violations.append(f"{lineno}: импорт {name!r} из {module!r}")
        elif name in FORBIDDEN_NAMES:
            # ``from recagent.models import Query`` и ``from recagent.agent import Agent``
            # попадают сюда же, если модуль не распознан как запрещённый целиком.
            violations.append(f"{lineno}: импорт запрещённого имени {name!r} из {module!r}")

    assert not violations, f"{path.name} нарушает независимость оракула:\n  " + "\n  ".join(violations)


def test_agent_code_does_not_import_theta():
    """Обратное направление: агент не должен иметь доступа к скрытым предпочтениям.

    Если ``recagent`` начнёт импортировать ``catalog.users``, theta перестанет быть
    скрытой, и success_rate снова станет измерением «насколько агент угадал то, что
    ему дали прочитать».
    """
    violations: list[str] = []
    for path in sorted((ROOT / "src" / "recagent").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for module, lineno, name in _imported_names(tree):
            if module.endswith("catalog.users") or module.endswith(".users"):
                violations.append(f"{path.relative_to(ROOT)}:{lineno}: импорт {module!r}")
            if name == "utility" and "recagent" in module:
                violations.append(f"{path.relative_to(ROOT)}:{lineno}: импорт utility из {module!r}")
    assert not violations, "код агента читает theta:\n  " + "\n  ".join(violations)


def _called_functions(tree: ast.AST) -> set[str]:
    """Все вызываемые имена файла. Имена, а не исходный текст: докстринг с
    упоминанием ``matches()`` не является вызовом и ловиться не должен."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
        elif isinstance(node, ast.Attribute):
            # Доступ без вызова: ``providers.matches`` как объект.
            names.add(node.attr)
    return names


@pytest.mark.parametrize("path", _evals_sources(), ids=lambda p: p.name)
def test_oracle_has_its_own_filter_implementation(path: Path):
    """Оракул не вызывает фильтры и объяснения агента ни прямо, ни через атрибут.

    Тест выше запрещает ИМПОРТ; этот запрещает ОБРАЩЕНИЕ, включая обход через
    локальное переименование, через ``module.func`` и через ``getattr`` с
    литеральным именем. Докстринги не в счёт: проверяется AST, а не текст.
    """
    called = _called_functions(ast.parse(path.read_text(encoding="utf-8")))
    forbidden_calls = called & {"matches", "rule_parse", "explain", "validate_evidence"}
    assert not forbidden_calls, f"{path.name} вызывает код агента: {sorted(forbidden_calls)}"


def test_oracle_defines_its_own_spoken_check():
    """Собственная реализация на месте, а не только «чужая не вызывается»."""
    tree = ast.parse((EVALS / "oracle.py").read_text(encoding="utf-8"))
    defined = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert "satisfies_spoken" in defined
    assert "judge" in defined


def test_judge_signature_excludes_agent_objects():
    """``judge`` принимает state и id, а не объект ответа агента.

    Дублирует тест в ``test_oracle.py`` на уровне сигнатуры, но здесь проверяет
    и ``build_criteria``: ни одна функция оракула не должна принимать
    ``ChatResponse`` или ``Query``.
    """
    import inspect

    import evals.oracle as oracle

    for name in ("judge", "build_criteria", "acceptable_set", "satisfies_spoken"):
        signature = inspect.signature(getattr(oracle, name))
        for parameter in signature.parameters.values():
            annotation = str(parameter.annotation)
            for forbidden in ("ChatResponse", "Query", "Agent", "Recommendation"):
                assert forbidden not in annotation, f"{name}({parameter.name}) принимает {forbidden}"
