"""Независимый оракул: критерий приёма ответа, который агент не может подделать.

Проблема, которую решает этот модуль
-------------------------------------
Прежняя оценка давала ``success_rate = 1.0``, потому что оракул проверял те же
поля, которые агент применял как жёсткие фильтры: система и критерий были написаны
друг под друга, и любая непустая выдача проходила по построению
(``docs/EVAL-PLAN.md`` §1). Прибор, который не может сказать «плохо», ничего не
измеряет.

Два независимых источника истины
--------------------------------
1. **theta** — скрытые предпочтения профиля (``recagent.catalog.users``). Агент их
   не видит никогда. Полезность объекта не выводится из того, что агент извлёк из
   реплики.
2. **spoken** — что пользователь ДЕЙСТВИТЕЛЬНО сказал вслух. Формируется
   генератором сценариев вместе с текстом реплик, до запуска агента. Это не
   ``Query`` агента: оракул не импортирует ``Query`` и не читает ``response.query``.

О независимости честно. ``SpokenConstraints`` использует те же имена слотов, что и
``Query``, — иначе и быть не может: слоты заданы доменом. Независимость здесь не в
«другой схеме», а в другом ИСТОЧНИКЕ (ground truth сценария против извлечённого
агентом) и в том, что ``satisfies_spoken`` написана заново, а не вызывает
``matches()``. Функция ``judge`` принимает только ``state`` и список id, а не
``ChatResponse``: имея объект ответа, слишком легко незаметно начать читать
``response.query``, а это и есть циркулярность.

Главное отличие от прежнего критерия: соответствие сказанным ограничениям
НЕОБХОДИМО, но НЕ ДОСТАТОЧНО. Объект обязан ещё попасть в приемлемое множество по
theta. Тест ``test_tautology_is_broken`` фиксирует именно это: выдача, формально
удовлетворяющая всем сказанным ограничениям, но theta не подходящая, проваливается.

Негативные контроли (``docs/DATA-PLAN.md`` §5.5) встроены в критерии, а не оставлены
на совести прогона:
  * приемлемое множество не может покрывать весь каталог — иначе успех тривиален;
  * оно не может быть пустым при достаточном множестве выполнимых ограничений —
    иначе сценарий непроходим по построению;
  * «потолок» B3 обязан проходить: лучший по theta ответ принимается. Если оракул
    отвергает и его, прибор сломан в другую сторону.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Literal

from pydantic import Field, field_validator

from recagent.catalog.users import Theta, utility
from recagent.models import Item, StrictModel

__all__ = [
    "ACCEPTABLE_QUANTILE",
    "MAX_ACCEPTABLE_SHARE",
    "MAX_ACCEPTABLE_SHARE_OF_FEASIBLE",
    "MIN_FEASIBLE_FOR_ACCEPTABLE",
    "ExpectedOutcome",
    "OracleCriteria",
    "SpokenConstraints",
    "Verdict",
    "acceptable_set",
    "build_criteria",
    "feasible_set",
    "is_degenerate",
    "iter_degenerate",
    "judge",
    "satisfies_spoken",
    "utility_threshold",
]

#: Квантиль полезности внутри множества выполнимых ограничений: приемлемыми
#: считаются объекты не ниже этой квантили. 0.75 = верхняя четверть.
#: Фиксировано ДО прогона экспериментов (``docs/EVAL-PLAN.md`` §5) и дублируется
#: в ``evals/thresholds.yaml``; расхождение между ними ловит тест.
ACCEPTABLE_QUANTILE: Final[float] = 0.75

#: Максимальная доля ВСЕГО каталога, которую может занимать приемлемое множество.
#: Страховка от сценария, где «успех» достижим любым ответом.
MAX_ACCEPTABLE_SHARE: Final[float] = 0.35

#: Максимальная доля ВЫПОЛНИМОГО множества, которую может занимать приемлемое.
#: Это рабочая проверка различающей силы порога. При квантили 0.75 доля равна
#: примерно 0.25; если она выросла к единице, порог перестал отбирать. Доля от
#: каталога такое вырождение НЕ ловит в принципе: она ограничена сверху
#: ``(1 - q) * feasible / catalog`` и при трёх выполнимых объектах равна 0.001,
#: то есть выглядит идеально.
MAX_ACCEPTABLE_SHARE_OF_FEASIBLE: Final[float] = 0.40

#: Минимальный размер множества выполнимых ограничений, при котором имеет смысл
#: строить приемлемое множество по квантили. На трёх объектах квантиль решает всё,
#: и разница между «прибор работает» и «прибор угадал» исчезает.
MIN_FEASIBLE_FOR_ACCEPTABLE: Final[int] = 8


class ExpectedOutcome(StrEnum):
    """Что оракул ждёт от диалога. Значения совпадают с ``state`` ответа агента.

    ``StrEnum``, а не ``(str, Enum)``: сравнение с обычной строкой остаётся
    (``ExpectedOutcome.RECOMMEND == "recommend"``), но ``str()`` даёт само
    значение, а не ``ExpectedOutcome.RECOMMEND`` — иначе коды провалов в отчёте
    выглядели бы как ``state_mismatch:ExpectedOutcome.RECOMMEND``.
    """

    RECOMMEND = "recommend"
    NO_RESULTS = "no_results"
    CLARIFY = "clarify"


class SpokenConstraints(StrictModel):
    """Что пользователь действительно сказал вслух — ground truth реплик.

    Заполняется генератором сценариев вместе с текстом, ДО запуска агента.
    Намеренно отдельная модель, а не ``recagent.models.Query``: у оракула не должно
    быть возможности прочитать то, что извлёк агент. ``None`` везде означает
    «об этом не говорили», а не «ограничение снято».
    """

    kind: Literal["series", "film", "course"] | None = None
    genre: str | None = None
    excluded_genres: list[str] = Field(default_factory=list, max_length=10)
    tone: str | None = None
    max_seasons: int | None = Field(default=None, ge=1, le=100)
    max_minutes: int | None = Field(default=None, ge=1, le=10000)
    level: str | None = None
    practical: bool | None = None

    @field_validator("genre", "tone", "level", "kind")
    @classmethod
    def _not_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("пустая строка вместо ограничения")
        return value


def satisfies_spoken(item: Item, spoken: SpokenConstraints) -> tuple[bool, tuple[str, ...]]:
    """Соответствует ли объект сказанным вслух ограничениям.

    Возвращает пару ``(выполнено, причины)``: причины нужны, чтобы провал
    диагностировался, а не угадывался. Это отдельная реализация, а не вызов
    ``recagent.providers.matches`` — иначе оракул снова проверял бы сам себя.

    Правило null: неизвестное значение атрибута НЕ удовлетворяет явному числовому
    ограничению (``max_seasons``) и явному требованию (``level``, ``practical``).
    Это семантика контракта, а не агента: платформа обязана различать «нет данных»
    и «совпало» (``docs/contract/openapi.yaml``).

    Расхождения с ``matches()`` намеренные и документированные:
      * ``max_seasons`` применяется к объектам, у которых сезоны есть, и к сериалам
        с неизвестным числом сезонов (для фильма ограничение неприменимо);
      * ``level`` проверяется как равенство любому объекту, а не только курсам,
        потому что сказанное вслух «начальный уровень» фильм удовлетворить не может.
    """
    reasons: list[str] = []

    if spoken.kind is not None and item.kind != spoken.kind:
        reasons.append("kind")
    if spoken.genre is not None and item.genre != spoken.genre:
        reasons.append("genre")
    if item.genre in spoken.excluded_genres:
        reasons.append("excluded_genre")
    if spoken.tone is not None and item.tone != spoken.tone:
        reasons.append("tone")
    if spoken.max_seasons is not None:
        if item.seasons is not None and item.seasons > spoken.max_seasons:
            reasons.append("max_seasons")
        elif item.kind == "series" and item.seasons is None:
            reasons.append("seasons_unknown")
    if spoken.max_minutes is not None and item.minutes > spoken.max_minutes:
        reasons.append("max_minutes")
    if spoken.level is not None and item.level != spoken.level:
        reasons.append("level")
    if spoken.practical is not None and item.practical != spoken.practical:
        reasons.append("practical")

    return not reasons, tuple(reasons)


def feasible_set(catalog: Sequence[Item], spoken: SpokenConstraints) -> list[Item]:
    """Объекты, удовлетворяющие сказанным ограничениям. Порядок каталога сохранён."""
    return [item for item in catalog if satisfies_spoken(item, spoken)[0]]


def utility_threshold(items: Sequence[Item], theta: Theta, quantile: float = ACCEPTABLE_QUANTILE) -> float:
    """Квантиль полезности по множеству. Пустое множество даёт бесконечность.

    Квантиль считается по фактическому распределению полезности ДАННОГО профиля на
    ДАННОМ подмножестве, а не берётся общей константой. Иначе у профиля с узкими
    вкусами (малая дисперсия полезности) порог оказался бы недостижим, а у
    «всеядного» — проходимым всем каталогом.
    """
    if not items:
        return math.inf
    values = sorted(utility(item, theta) for item in items)
    if len(values) == 1:
        return values[0]
    # Ближайший ранг, как в numpy по умолчанию: index = ceil(q*n) - 1.
    index = max(0, min(len(values) - 1, math.ceil(quantile * len(values)) - 1))
    return values[index]


def acceptable_set(
    catalog: Sequence[Item], theta: Theta, spoken: SpokenConstraints, quantile: float = ACCEPTABLE_QUANTILE
) -> frozenset[str]:
    """Приемлемое множество: выполнил сказанное И достаточно хорош по theta.

    Это и есть критерий успеха. Множество строится от theta, а не от того, что
    извлёк агент, поэтому его нельзя «подогнать» качеством парсинга.
    """
    feasible = feasible_set(catalog, spoken)
    threshold = utility_threshold(feasible, theta, quantile)
    return frozenset(item.id for item in feasible if utility(item, theta) >= threshold)


@dataclass(frozen=True, kw_only=True)
class OracleCriteria:
    """Полный критерий одного сценария. Строится ДО диалога и в нём не меняется.

    Неизменяемость принципиальна: критерий, который можно поправить по ходу
    прогона, — это не критерий. Поэтому ``frozen=True``, а все множества
    вычисляются в ``build_criteria``.
    """

    user_id: str
    theta: Theta
    spoken: SpokenConstraints
    expected: ExpectedOutcome
    acceptable_ids: frozenset[str]
    threshold: float
    catalog_size: int
    max_clarifications: int
    feasible_count: int
    #: Объекты, из которых состоит «потолок» B3: лучшие по utility среди
    #: выполнимых. Нужны, чтобы нормировать улучшение и проверить сам прибор.
    ceiling_ids: tuple[str, ...] = ()

    @property
    def acceptable_share(self) -> float:
        """Доля всего каталога, которую занимает приемлемое множество."""
        return len(self.acceptable_ids) / self.catalog_size if self.catalog_size else 0.0

    @property
    def acceptable_share_of_feasible(self) -> float:
        """Доля ВЫПОЛНИМОГО множества, которую занимает приемлемое.

        Именно эта величина показывает, отбирает ли порог что-либо. Ожидаемое
        значение — ``1 - ACCEPTABLE_QUANTILE`` (0.25); рост к единице означает,
        что порог выродился в «принять всё».
        """
        return len(self.acceptable_ids) / self.feasible_count if self.feasible_count else 0.0


def build_criteria(
    *,
    catalog: Sequence[Item],
    theta: Theta,
    user_id: str,
    spoken: SpokenConstraints,
    expected: ExpectedOutcome | None = None,
    patience: int | None = None,
    quantile: float = ACCEPTABLE_QUANTILE,
    ceiling_size: int = 5,
) -> OracleCriteria:
    """Собрать критерий сценария.

    ``expected`` выводится из данных, если не задан явно: нет выполнимых объектов —
    честный ответ «не найдено», а не выдача чего попало. Ожидание CLARIFY задаётся
    только сценарием: оракул не может угадать, достаточно ли информации в первой
    реплике, — это решение генератора сценариев.
    """
    if not catalog:
        raise ValueError("пустой каталог: критерий не построить")

    feasible = feasible_set(catalog, spoken)
    ranked = sorted(feasible, key=lambda item: (-utility(item, theta), item.id))
    threshold = utility_threshold(feasible, theta, quantile)
    acceptable = frozenset(item.id for item in feasible if utility(item, theta) >= threshold)

    if expected is None:
        expected = ExpectedOutcome.RECOMMEND if feasible else ExpectedOutcome.NO_RESULTS

    return OracleCriteria(
        user_id=user_id,
        theta=theta,
        spoken=spoken,
        expected=expected,
        acceptable_ids=acceptable,
        threshold=threshold,
        catalog_size=len(catalog),
        max_clarifications=patience if patience is not None else theta.patience,
        feasible_count=len(feasible),
        ceiling_ids=tuple(item.id for item in ranked[:ceiling_size]),
    )


@dataclass(frozen=True, kw_only=True)
class Verdict:
    """Решение оракула по одному ответу агента.

    ``failures`` — машинно-читаемые коды, а не свободный текст: по ним строится
    статистика причин провалов в отчёте (``evals/metrics.py``).
    """

    success: bool
    expected: ExpectedOutcome
    shown_ids: tuple[str, ...]
    hits: tuple[str, ...]
    misses: tuple[str, ...]
    failures: tuple[str, ...]
    mean_utility: float | None
    best_utility: float | None

    @property
    def hit_rate(self) -> float | None:
        """Доля приемлемых объектов в выдаче. None, если выдачи не было."""
        if not self.shown_ids:
            return None
        return len(self.hits) / len(self.shown_ids)

    def as_dict(self) -> dict[str, object]:
        return {
            "success": self.success,
            "expected": self.expected.value,
            "shown": list(self.shown_ids),
            "hits": list(self.hits),
            "misses": list(self.misses),
            "failures": list(self.failures),
            "hit_rate": self.hit_rate,
            "mean_utility": self.mean_utility,
            "best_utility": self.best_utility,
        }


def judge(
    *,
    criteria: OracleCriteria,
    catalog_by_id: dict[str, Item],
    state: str,
    shown_ids: Sequence[str],
    clarifications: int = 0,
    require_all_acceptable: bool = True,
) -> Verdict:
    """Вынести решение по ответу агента.

    ``require_all_acceptable`` — строгость критерия выдачи. По умолчанию ВСЕ
    показанные объекты обязаны быть приемлемыми: пользователь видит весь список, а
    не только первую строку, и один неподходящий объект в выдаче из пяти — это
    испорченная рекомендация, а не «80% успеха». Мягкий режим (хотя бы один) нужен
    для сравнения конфигураций и включается явно.
    """
    failures: list[str] = []
    if any(item_id not in catalog_by_id for item_id in shown_ids):
        # Галлюцинация в строгом смысле: объект, которого нет в каталоге.
        failures.append("unknown_item_id")

    hits = tuple(item_id for item_id in shown_ids if item_id in criteria.acceptable_ids)
    misses = tuple(item_id for item_id in shown_ids if item_id in catalog_by_id and item_id not in criteria.acceptable_ids)

    utilities = [utility(catalog_by_id[item_id], criteria.theta) for item_id in shown_ids if item_id in catalog_by_id]
    mean_utility = round(math.fsum(utilities) / len(utilities), 6) if utilities else None
    best_utility = round(max(utilities), 6) if utilities else None

    if state != criteria.expected.value:
        failures.append(f"state_mismatch:{state}")

    if criteria.expected is ExpectedOutcome.RECOMMEND:
        if not shown_ids:
            failures.append("empty_recommendation")
        elif require_all_acceptable and misses:
            failures.append("unacceptable_item_shown")
        elif not require_all_acceptable and not hits:
            failures.append("no_acceptable_item")
    elif criteria.expected is ExpectedOutcome.NO_RESULTS:
        # Пустая выдача недостаточна: агент обязан СОХРАНИТЬ ограничения, а не
        # ослабить их и выдать что попало. Поэтому проверяем факт выдачи.
        if shown_ids:
            failures.append("results_where_none_exist")
    elif criteria.expected is ExpectedOutcome.CLARIFY and shown_ids:
        failures.append("answered_instead_of_clarifying")

    if clarifications > criteria.max_clarifications:
        failures.append("patience_exceeded")

    return Verdict(
        success=not failures,
        expected=criteria.expected,
        shown_ids=tuple(shown_ids),
        hits=hits,
        misses=misses,
        failures=tuple(failures),
        mean_utility=mean_utility,
        best_utility=best_utility,
    )


def is_degenerate(criteria: OracleCriteria) -> tuple[bool, tuple[str, ...]]:
    """Признак вырожденного сценария.

    Отдельная функция, а не исключение внутри ``build_criteria``: вырожденность —
    свойство НАБОРА сценариев, и видеть её нужно сводно (сколько процентов набора
    непригодно и почему), а не ловить по одному исключению в цикле.

    Четыре признака, и все независимы:
      * ``empty_acceptable_set`` — принять нечего, сценарий непроходим;
      * ``feasible_too_small`` — квантиль на малом множестве неустойчива;
      * ``acceptable_covers_feasible`` — порог перестал отбирать;
      * ``acceptable_covers_catalog`` — успех достижим любым ответом.

    Сценарий с ``expected=NO_RESULTS`` не проверяется: у него приемлемое множество
    пусто по определению, и это честная пустота, а не вырождение.
    """
    if criteria.expected is not ExpectedOutcome.RECOMMEND:
        return False, ()

    problems: list[str] = []
    if not criteria.acceptable_ids:
        problems.append("empty_acceptable_set")
    if criteria.feasible_count < MIN_FEASIBLE_FOR_ACCEPTABLE:
        problems.append(f"feasible_too_small:{criteria.feasible_count}")
    if criteria.acceptable_share_of_feasible > MAX_ACCEPTABLE_SHARE_OF_FEASIBLE:
        problems.append(f"acceptable_covers_feasible:{criteria.acceptable_share_of_feasible:.3f}")
    if criteria.acceptable_share > MAX_ACCEPTABLE_SHARE:
        problems.append(f"acceptable_covers_catalog:{criteria.acceptable_share:.3f}")
    return bool(problems), tuple(problems)


def iter_degenerate(criteria: Iterable[OracleCriteria]) -> list[tuple[str, tuple[str, ...]]]:
    """Все вырожденные сценарии набора: ``(user_id, причины)``."""
    return [(item.user_id, problems) for item in criteria if (problems := is_degenerate(item)[1])]
