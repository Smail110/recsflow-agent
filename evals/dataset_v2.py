"""Versioned, frozen evaluation datasets with explicit split-leakage checks.

Public ``dev`` and ``validation`` data are synthetic and reproducible.  A
``final_holdout`` is deliberately loader-only: it must be supplied by an
independent source outside the repository and is protected by the runner's
one-time consumption guard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Final, Literal

from pydantic import Field, model_validator

from evals.oracle import ExpectedOutcome, OracleCriteria, SpokenConstraints, build_criteria, is_degenerate
from evals.scenarios import ADVERSARIAL_KINDS, BUILDERS, SCENARIO_KINDS, OracleCriteriaSpec, Scenario, Turn
from recagent.catalog import catalog_sha256, generate_catalog
from recagent.catalog.users import UserProfile, generate_profiles
from recagent.models import StrictModel

ROOT: Final = Path(__file__).resolve().parents[1]
DATASET_VERSION: Final = "recagent-eval-v2.0"
PUBLIC_SPLITS: Final = ("dev", "validation")
SplitV2 = Literal["dev", "validation", "final_holdout"]

# Different catalog, profile and dialogue streams prevent accidental reuse.
SEEDS: Final[dict[str, dict[str, int]]] = {
    "dev": {"catalog": 4202, "profiles": 4203, "scenarios": 4204},
    "validation": {"catalog": 7301, "profiles": 7302, "scenarios": 7303},
}

KIND_WORD = {"series": "сериал", "film": "фильм", "course": "курс"}


class V2Scenario(Scenario):
    split: SplitV2


class V2Case(StrictModel):
    dataset_version: str = DATASET_VERSION
    case_id: str
    scenario_family: str
    surface_family_id: str
    surface_pattern: str
    synthetic: bool = True
    human_authored: bool = False
    scenario: V2Scenario

    @model_validator(mode="after")
    def consistent(self) -> V2Case:
        if self.case_id != self.scenario.scenario_id:
            raise ValueError("case_id and scenario_id differ")
        if self.scenario_family != self.scenario.kind:
            raise ValueError("scenario_family and scenario.kind differ")
        return self


class V2Manifest(StrictModel):
    schema_version: int = 2
    dataset_version: str = DATASET_VERSION
    split: SplitV2
    origin: Literal["synthetic_code", "independent_external"]
    blind_claim: bool = False
    cases_file: str
    qa_sample_file: str
    cases_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    case_count: int = Field(gt=0)
    unique_users: int = Field(gt=0)
    catalog_seed: int = Field(ge=0)
    catalog_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    profile_seed: int = Field(ge=0)
    scenario_seed: int = Field(ge=0)
    by_family: dict[str, int]
    surface_families: list[str]
    normalized_dialogues_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    statistical_target: dict[str, bool | float | int | list[float]]
    limitations: list[str]


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_value(value: object) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def normalize_text(text: str) -> str:
    text = text.casefold().replace("ё", "е")
    text = re.sub(r"[^a-zа-я0-9]+", " ", text)
    return " ".join(text.split())


def normalized_dialogue(case: V2Case) -> str:
    return " || ".join(normalize_text(turn.utterance) for turn in case.scenario.turns)


def _manifest_payload(manifest: V2Manifest | dict) -> dict:
    payload = manifest.model_dump(mode="json") if isinstance(manifest, V2Manifest) else dict(manifest)
    payload.pop("manifest_sha256", None)
    return payload


def _spec(criteria: OracleCriteria) -> OracleCriteriaSpec:
    return OracleCriteriaSpec(
        acceptable_ids=sorted(criteria.acceptable_ids),
        ceiling_ids=list(criteria.ceiling_ids),
        threshold=criteria.threshold,
        catalog_size=criteria.catalog_size,
        feasible_count=criteria.feasible_count,
        max_clarifications=criteria.max_clarifications,
    )


def _assemble(
    *, case_id: str, kind: str, split: str, seed: int, profile: UserProfile, turns: list[Turn], catalog: Sequence
) -> Scenario | None:
    final = turns[-1]
    criteria = build_criteria(
        catalog=catalog,
        theta=profile.theta,
        user_id=profile.user_id,
        spoken=final.spoken,
        expected=final.expected,
        patience=profile.theta.patience,
    )
    if is_degenerate(criteria)[0]:
        return None
    after = None
    if final.answer_if_clarified is not None and final.spoken_after_answer is not None:
        second = build_criteria(
            catalog=catalog,
            theta=profile.theta,
            user_id=profile.user_id,
            spoken=final.spoken_after_answer,
            expected=ExpectedOutcome.RECOMMEND,
            patience=profile.theta.patience,
        )
        if is_degenerate(second)[0]:
            return None
        after = _spec(second)
    return V2Scenario(
        scenario_id=case_id,
        kind=kind,
        split=split,
        seed=seed,
        user_id=profile.user_id,
        theta=profile.theta,
        turns=turns,
        criteria=_spec(criteria),
        criteria_after_clarify=after,
        feasible_count=criteria.feasible_count,
        acceptable_count=len(criteria.acceptable_ids),
        adversarial_note=(
            "Синтетический adversarial-сценарий v2; проверяет отрицание, противоречие или честный пустой результат."
            if kind in ADVERSARIAL_KINDS
            else None
        ),
    )


def _value(spoken: SpokenConstraints, name: str, default: object = "") -> object:
    value = getattr(spoken, name)
    return default if value is None else value


def _explicit_clauses(spoken: SpokenConstraints) -> tuple[list[str], list[str]]:
    """Ground-truth clauses and their placeholder patterns, slot by slot."""
    clauses: list[str] = []
    patterns: list[str] = []

    def add(text: str, pattern: str) -> None:
        clauses.append(text)
        patterns.append(pattern)

    if spoken.kind is not None:
        add(f"формат — {KIND_WORD[spoken.kind]}", "формат — {kind}")
    if spoken.genre is not None:
        add(f"жанр — {spoken.genre}", "жанр — {genre}")
    if spoken.tone is not None:
        add(f"тон — {spoken.tone}", "тон — {tone}")
    for genre in spoken.excluded_genres:
        add(f"исключить жанр {genre}", "исключить жанр {excluded_genre}")
    for tone in spoken.excluded_tones:
        add(f"исключить тон {tone}", "исключить тон {excluded_tone}")
    if spoken.max_minutes is not None:
        add(f"не дольше {spoken.max_minutes} минут", "не дольше {minutes} минут")
    if spoken.max_seasons is not None:
        add(f"не больше {spoken.max_seasons} сезонов", "не больше {seasons} сезонов")
    if spoken.level is not None:
        add(f"уровень — {spoken.level}", "уровень — {level}")
    if spoken.practical is True:
        add("обязательно с практическими заданиями", "обязательно с практическими заданиями")
    elif spoken.practical is False:
        add("без практических заданий, только теория", "без практических заданий, только теория")
    if spoken.named_title is not None:
        add(f"точное название — «{spoken.named_title}»", "точное название — «{title}»")
    if spoken.seed_title is not None:
        add(f"похожее на «{spoken.seed_title}», но не оно само", "похожее на «{title}», но не оно само")
    return clauses, patterns


def _describe_pattern(split: str, variant: int, spoken: SpokenConstraints) -> tuple[str, str]:
    """Render every explicit slot without defaults or calls to the product parser."""
    clauses, patterns = _explicit_clauses(spoken)
    if not clauses:
        raise ValueError("generic renderer cannot verbalize an empty constraint set")
    grammar = variant % 5
    joined = "; ".join(clauses)
    pattern_joined = "; ".join(patterns)
    if split == "dev":
        cores = (
            "Подберите вариант: {details}.",
            "Требования к рекомендации: {details}.",
            "Ищу по таким условиям: {details}.",
            "Что можно предложить, если {details}?",
            "Запрос к каталогу следующий: {details}.",
        )
    else:
        cores = (
            "Накидай что-нибудь, вводные такие: {details}.",
            "Мне бы вариант; по условиям вот так: {details}.",
            "Есть что подходящее? Нужно, чтобы {details}.",
            "Отсекай всё лишнее: {details}.",
            "Запрос по-простому: {details}.",
        )
    return cores[grammar].format(details=joined), cores[grammar].format(details=pattern_joined)


def _simple_patterns(split: str, kind: str) -> tuple[str, ...]:
    tables = {
        "dev": {
            "discovery_vague": (
                "Посоветуйте что-нибудь достойное.",
                "Помогите выбрать, пока без конкретных условий.",
                "Не знаю, что включить; задайте уточняющий вопрос.",
                "Хочу рекомендацию, но ещё не определился с форматом.",
                "Подберите вариант; детали могу уточнить.",
            ),
            "discovery_genre": (
                "Посоветуйте что-нибудь: жанр — {genre}.",
                "Ищу рекомендацию, интересующий жанр — {genre}.",
                "Что достойного есть, если выбрать жанр {genre}?",
                "Нужен жанр {genre}, формат пока не важен.",
                "Подберите вариант: интересует жанр {genre}.",
            ),
            "mood_evening": (
                "На вечер хочется чего-то {tone}.",
                "Настроение сегодня — {tone}; что посмотреть?",
                "Подберите что-нибудь с тоном «{tone}».",
                "Без жанровых требований, главное чтобы {tone}.",
                "Ищу вариант на вечер: настроение {tone}.",
            ),
        },
        "validation": {
            "discovery_vague": (
                "Что бы такое заценить? Сам не решил.",
                "Подкинь идею, если надо — спроси подробности.",
                "Залипнуть бы во что-нибудь, но вводных пока ноль.",
                "Выбрать нечего; сориентируй вопросом.",
                "Хочу рекомендацию вслепую, уточняй важное.",
            ),
            "discovery_genre": (
                "Тянет на жанр {genre}; чем заняться?",
                "Накидай чего-нибудь, жанр {genre}, без привязки к формату.",
                "Сегодня жанровый вайб — {genre}. Есть идеи?",
                "По жанру хочу {genre}, в остальном открыт.",
                "Если жанр — {genre}, что посоветуешь?",
            ),
            "mood_evening": (
                "Сегодня хочется вайб {tone}; жанр не принципиален.",
                "Что-нибудь бы {tone} на вечерок.",
                "Формат любой, лишь бы по настроению {tone}.",
                "Подкинь вариант: сейчас настрой {tone}.",
                "Есть что глянуть под {tone} настроение?",
            ),
        },
    }
    return tables[split][kind]


def _speech_frame(split: str, variation: int, core: str) -> tuple[str, int]:
    """Apply genuinely different request constructions, not numeric suffixes."""
    if split == "dev":
        frames = (
            "{core}",
            "Пожалуйста: {core}",
            "Сформулирую запрос так: {core}",
            "Можете помочь? {core}",
            "Для сегодняшнего выбора: {core}",
            "У меня конкретное условие — {core}",
            "Подскажите по каталогу: {core}",
            "Хочу подобрать заранее. {core}",
            "Буду признателен за совет: {core}",
            "Если есть подходящее, то {core}",
        )
    else:
        frames = (
            "{core}",
            "Короче, вот что нужно: {core}",
            "Смотри, ситуация такая — {core}",
            "Закинь идею: {core}",
            "Есть просьба на сейчас. {core}",
            "Не мудрствуя: {core}",
            "Можно быстро подобрать? {core}",
            "Я тут выбираю, и запрос такой: {core}",
            "По-простому говоря, {core}",
            "Если найдётся — супер: {core}",
        )
    frame_index = (variation // 5) % len(frames)
    return frames[frame_index].format(core=core), frame_index


def _render_turns(split: str, kind: str, turns: list[Turn], variation: int) -> tuple[list[Turn], str, str]:
    """Return turns, a macro-family id and the concrete placeholder pattern."""
    rendered: list[Turn] = []
    patterns: list[str] = []
    core_patterns: list[str] = []
    for index, turn in enumerate(turns):
        spoken = turn.spoken
        v = variation + index * 3
        pattern: str
        values = {
            "kind": KIND_WORD.get(str(_value(spoken, "kind")), "вариант"),
            "genre": str(_value(spoken, "genre", "любой жанр")),
            "tone": str(_value(spoken, "tone", "нейтральный")),
            "minutes": str(_value(spoken, "max_minutes", "без лимита")),
            "seasons": str(_value(spoken, "max_seasons", "без лимита")),
            "level": str(_value(spoken, "level", "любой")),
            "title": str(_value(spoken, "named_title", _value(spoken, "seed_title", ""))),
            "excluded": ", ".join(spoken.excluded_genres),
        }
        if kind in {"discovery_vague", "discovery_genre", "mood_evening"} and (kind != "discovery_vague" or index == 0):
            choices = _simple_patterns(split, kind)
            pattern = choices[v % len(choices)]
        elif kind == "discovery_vague":
            choices = _simple_patterns(split, "discovery_genre")
            pattern = choices[v % len(choices)]
        elif kind in {"discovery_multi", "release_constraint", "domain_switch"} and index == 0:
            _, pattern = _describe_pattern(split, v, spoken)
        elif kind == "incremental":
            if index == 0:
                pattern = (
                    "Ищу вариант: формат — {kind}, жанр — {genre}."
                    if split == "dev"
                    else "Для начала накинь вариант: жанр — {genre}, формат — {kind}."
                )
            else:
                pattern = (
                    "Добавлю условие: не дольше {minutes} минут." if split == "dev" else "А, ещё момент: всё длиннее {minutes} минут мимо."
                )
        elif kind == "release_constraint" and index == 1:
            if turns[0].spoken.max_seasons is not None:
                pattern = "Ограничение по числу сезонов снимите." if split == "dev" else "Ладно, сколько там сезонов — уже без разницы."
            else:
                pattern = (
                    "Ограничение по длительности больше не учитывать."
                    if split == "dev"
                    else "Передумал: хронометраж теперь вообще не важен."
                )
        elif kind == "domain_switch" and index == 1:
            pattern = (
                "Сменим задачу: нужен курс по теме «{genre}»."
                if split == "dev"
                else "Стоп, кино отменяется — давай лучше курс по теме «{genre}»."
            )
        elif kind == "similar_seed":
            pattern = (
                "Подберите {kind}, похожий на «{title}», но не сам этот объект."
                if split == "dev"
                else "Хочу {kind} в духе «{title}»; оригинал повторно не предлагай."
            )
        elif kind == "navigation_title":
            pattern = (
                "Найдите в каталоге точное название «{title}»."
                if split == "dev"
                else "Проверь, есть ли у вас прямо «{title}» — нужен именно он."
            )
        elif kind in {"course_level", "course_practical"}:
            if spoken.practical:
                choices = (
                    (
                        "Нужен курс по теме «{genre}», обязательно с практическими заданиями.",
                        "Подберите курс по теме «{genre}»: теория должна сопровождаться практическими заданиями.",
                        "Хочу изучать тему «{genre}» на курсе с практическими заданиями.",
                        "Курс по теме «{genre}» подойдёт только с практическими заданиями.",
                        "Ищу курс по теме «{genre}»; обязательное условие — практические задания.",
                    )
                    if split == "dev"
                    else (
                        "Ищу курс по теме «{genre}»: обязательно с практическими заданиями, одной теории мало.",
                        "По теме «{genre}» нужен курс, где есть практические задания, а не одна болтовня.",
                        "Хочу прокачать тему «{genre}» руками — курс давай с практическими заданиями.",
                        "Накидай курс по теме «{genre}»; без практических заданий даже не предлагай.",
                        "Нужен курс по теме «{genre}» с упором на практические задания.",
                    )
                )
            else:
                choices = (
                    (
                        "Подберите курс по теме «{genre}», уровень {level}.",
                        "Ищу курс по теме «{genre}», рассчитанный на уровень {level}.",
                        "Нужен курс по теме «{genre}»; мой уровень — {level}.",
                        "Курс по теме «{genre}» должен соответствовать уровню {level}.",
                        "Посоветуйте курс по теме «{genre}» для уровня {level}.",
                    )
                    if split == "dev"
                    else (
                        "Хочу зайти в тему «{genre}»; курс нужен для уровня «{level}».",
                        "Какой курс взять по теме «{genre}», если уровень сейчас {level}?",
                        "Ищу курс по теме «{genre}»; по сложности мой уровень {level}.",
                        "Накидай курс по теме «{genre}», ориентир по уровню — {level}.",
                        "С курсом по теме «{genre}» не усложняй: целевой уровень {level}.",
                    )
                )
            pattern = choices[v % len(choices)]
        elif kind == "exclusion_genre":
            pattern = (
                "Подберите вариант: формат — {kind}, жанр — {genre}; исключите жанр {excluded}."
                if split == "dev"
                else "Хочу вариант: жанр — {genre}, формат — {kind}; только без жанра {excluded}."
            )
        elif kind == "tone_negation":
            pattern = (
                "Нужен вариант: формат — {kind}, жанр — {genre}; исключить тон мрачный."
                if split == "dev"
                else "Накидай вариант: жанр — {genre}, формат — {kind}; тон мрачный сразу исключаем."
            )
        elif kind == "contradiction":
            pattern = (
                "Нужен формат — {kind}; жанр {genre} одновременно выберите и исключите."
                if split == "dev"
                else "Хочу формат — {kind} и жанр {genre}, хотя жанр {genre} терпеть не могу — помоги разрулить."
            )
        elif kind == "no_result":
            pattern = (
                "Найдите вариант: формат — {kind}, жанр — {genre}, длительность не более 1 минуты (60 секунд)."
                if split == "dev"
                else "Нужен вариант: жанр — {genre}, формат — {kind}; уложиться в 1 минуту, то есть 60 секунд."
            )
        elif kind == "more_variants":
            pattern = (
                ("Подберите вариант: формат — {kind}, жанр — {genre}." if index == 0 else "Покажите другие варианты без повторов.")
                if split == "dev"
                else ("Накидай вариант: жанр — {genre}, формат — {kind}." if index == 0 else "Давай ещё, уже показанное не повторяй.")
            )
        else:
            _, pattern = _describe_pattern(split, v, spoken)
        core_patterns.append(pattern)
        pattern, _ = _speech_frame(split, v, pattern)
        text = pattern.format(**values)
        rendered.append(turn.model_copy(update={"utterance": text}))
        patterns.append(pattern)
    family = f"{kind}:{sha256_value(core_patterns)[:16]}"
    return rendered, family, " || ".join(patterns)


def generate_public_dataset(split: Literal["dev", "validation"], *, per_family: int = 25) -> tuple[list[V2Case], dict]:
    if split not in PUBLIC_SPLITS or per_family < 1:
        raise ValueError("public split must be dev/validation and per_family positive")
    seeds = SEEDS[split]
    catalog = generate_catalog(seeds["catalog"])
    target = per_family * len(SCENARIO_KINDS)
    profiles = generate_profiles(seeds["profiles"], target + max(400, target // 3), catalog)
    rng = random.Random(seeds["scenarios"])
    cases: list[V2Case] = []
    rejected: Counter[str] = Counter()
    seen_dialogues: set[str] = set()
    profile_index = 0
    for kind in SCENARIO_KINDS:
        accepted = 0
        attempts = 0
        while accepted < per_family:
            attempts += 1
            if attempts > per_family * 30 or profile_index >= len(profiles):
                raise RuntimeError(f"capacity exhausted for {kind}: {accepted}/{per_family}")
            raw_profile = profiles[profile_index]
            profile_index += 1
            profile = raw_profile.model_copy(update={"user_id": f"v2-{split}-{raw_profile.user_id}"})
            turns, _ = BUILDERS[kind](rng, profile.theta, catalog, profile)  # type: ignore[operator]
            variation = accepted + attempts * 7
            turns, surface_family, surface_pattern = _render_turns(split, kind, turns, variation)
            case_id = f"v2-{kind}-{split}-{accepted + 1:04d}"
            scenario = _assemble(
                case_id=case_id,
                kind=kind,
                split=split,
                seed=seeds["scenarios"],
                profile=profile,
                turns=turns,
                catalog=catalog,
            )
            if scenario is None:
                rejected[f"{kind}:degenerate"] += 1
                continue
            case = V2Case(
                case_id=case_id,
                scenario_family=kind,
                surface_family_id=surface_family,
                surface_pattern=surface_pattern,
                scenario=scenario,
            )
            normalized = normalized_dialogue(case)
            if normalized in seen_dialogues:
                rejected[f"{kind}:duplicate_dialogue"] += 1
                continue
            seen_dialogues.add(normalized)
            cases.append(case)
            accepted += 1
    validate_cases(cases, expected_split=split)
    return cases, {
        "attempted_profiles": profile_index,
        "rejected": dict(sorted(rejected.items())),
        "capacity": len(cases),
        "target": target,
    }


def semantic_surface_problems(case: V2Case) -> list[str]:
    """Check spoken ground truth against cumulative text without product parsing."""
    problems: list[str] = []
    said = ""
    for turn in case.scenario.turns:
        said = f"{said} {normalize_text(turn.utterance)}"
        spoken = turn.spoken

        def require(value: object, field: str, said_text: str = said, turn_index: int = turn.index) -> None:
            if normalize_text(str(value)) not in said_text:
                problems.append(f"turn {turn_index}: {field}={value!r} is not verbalized")

        if spoken.kind is not None:
            require(KIND_WORD[spoken.kind], "kind")
        if spoken.genre is not None:
            require(spoken.genre, "genre")
        if spoken.tone is not None:
            require(spoken.tone, "tone")
        for value in spoken.excluded_genres:
            require(value, "excluded_genres")
        for value in spoken.excluded_tones:
            require(value, "excluded_tones")
        if spoken.max_minutes is not None:
            require(spoken.max_minutes, "max_minutes")
            if "минут" not in said and "секунд" not in said:
                problems.append(f"turn {turn.index}: max_minutes has no time unit")
        if spoken.max_seasons is not None:
            require(spoken.max_seasons, "max_seasons")
            if "сезон" not in said:
                problems.append(f"turn {turn.index}: max_seasons has no season unit")
        if spoken.level is not None:
            require(spoken.level, "level")
        if spoken.named_title is not None:
            require(spoken.named_title, "named_title")
        if spoken.seed_title is not None:
            require(spoken.seed_title, "seed_title")
        if spoken.practical is True and "практическ" not in said:
            problems.append(f"turn {turn.index}: practical=true is not verbalized")
        if spoken.practical is False and not ("без" in said.split() and "практическ" in said):
            problems.append(f"turn {turn.index}: practical=false is not verbalized")
    final_text = normalize_text(case.scenario.turns[-1].utterance)
    if case.scenario_family == "release_constraint" and not any(
        marker in final_text for marker in ("снимите", "без разницы", "не учитывать", "не важен")
    ):
        problems.append("release_constraint does not explicitly remove the constraint")
    if case.scenario_family == "domain_switch" and not any(marker in final_text for marker in ("сменим", "отменяется", "лучше курс")):
        problems.append("domain_switch does not explicitly switch domain")
    if case.scenario_family == "more_variants" and not any(
        marker in final_text for marker in ("другие", "без повторов", "еще", "не повторяй")
    ):
        problems.append("more_variants does not request unseen items")
    return problems


def validate_cases(cases: Sequence[V2Case], *, expected_split: str | None = None) -> None:
    if not cases:
        raise ValueError("dataset is empty")
    ids = [case.case_id for case in cases]
    users = [case.scenario.user_id for case in cases]
    dialogues = [normalized_dialogue(case) for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case ids")
    if len(users) != len(set(users)):
        raise ValueError("one case per independent user is required")
    if len(dialogues) != len(set(dialogues)):
        raise ValueError("exact normalized dialogue duplicates")
    if any(case.dataset_version != DATASET_VERSION for case in cases):
        raise ValueError("mixed dataset versions")
    semantic_problems = {case.case_id: issues for case in cases if (issues := semantic_surface_problems(case))}
    if semantic_problems:
        first = next(iter(semantic_problems.items()))
        raise ValueError(f"surface text loses ground truth: {first}")
    if expected_split is not None and any(case.scenario.split != expected_split for case in cases):
        raise ValueError("scenario split disagrees with manifest split")


def _tokens(text: str) -> frozenset[str]:
    words = normalize_text(text).split()
    return frozenset(" ".join(words[index : index + 2]) for index in range(max(1, len(words) - 1)))


def near_duplicate_pairs(left: Sequence[V2Case], right: Sequence[V2Case], *, threshold: float = 0.88) -> list[tuple[str, str, float]]:
    """Near duplicates of placeholder patterns, not value-filled examples."""
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    left_patterns = {case.surface_pattern: _tokens(case.surface_pattern) for case in left}
    right_patterns = {case.surface_pattern: _tokens(case.surface_pattern) for case in right}
    found: list[tuple[str, str, float]] = []
    for a, a_tokens in left_patterns.items():
        for b, b_tokens in right_patterns.items():
            union = a_tokens | b_tokens
            score = len(a_tokens & b_tokens) / len(union) if union else 1.0
            if score >= threshold:
                found.append((sha256_value(a)[:12], sha256_value(b)[:12], round(score, 4)))
    return found


def validate_split_isolation(left: Sequence[V2Case], right: Sequence[V2Case], *, near_threshold: float = 0.88) -> dict:
    validate_cases(left)
    validate_cases(right)
    user_overlap = {case.scenario.user_id for case in left} & {case.scenario.user_id for case in right}
    family_overlap = {case.surface_family_id for case in left} & {case.surface_family_id for case in right}
    pattern_overlap = {normalize_text(case.surface_pattern) for case in left} & {normalize_text(case.surface_pattern) for case in right}
    dialogue_overlap = {normalized_dialogue(case) for case in left} & {normalized_dialogue(case) for case in right}
    near = near_duplicate_pairs(left, right, threshold=near_threshold)
    problems = {
        "users": len(user_overlap),
        "surface_families": len(family_overlap),
        "patterns": len(pattern_overlap),
        "dialogues": len(dialogue_overlap),
        "near_patterns": len(near),
    }
    if any(problems.values()):
        raise ValueError(f"split leakage detected: {problems}; near={near[:5]}")
    return {**problems, "near_threshold": near_threshold}


def statistical_target() -> dict[str, bool | float | int | list[float]]:
    return {
        "independent_profiles": 1600,
        "minimum_per_primary_family": 100,
        "alpha_two_sided": 0.05,
        "power": 0.80,
        "mde": 0.05,
        "paired_discordance_main": 0.20,
        "paired_discordance_sensitivity": [0.10, 0.30, 0.50],
        "absolute_rate_wilson_half_width": 0.03,
    }


def write_dataset(cases: Sequence[V2Case], output_dir: Path, *, split: Literal["dev", "validation"]) -> tuple[Path, Path]:
    validate_cases(cases, expected_split=split)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_path = output_dir / f"{DATASET_VERSION}-{split}.jsonl"
    payload = "".join(canonical_json(case.model_dump(mode="json")) + "\n" for case in cases).encode("utf-8")
    cases_path.write_bytes(payload)
    counts = Counter(case.scenario_family for case in cases)
    qa_path = output_dir / f"{DATASET_VERSION}-{split}.qa-sample.md"
    sample_by_family = {family: next(case for case in cases if case.scenario_family == family) for family in sorted(counts)}
    qa_lines = [
        f"# QA-выборка {DATASET_VERSION}: {split}",
        "",
        "Синтетические примеры для ручной проверки. Чекбоксы намеренно не отмечены автоматически.",
        "",
    ]
    for family, case in sample_by_family.items():
        qa_lines.extend([f"## {family}", "", "- [ ] Смысл реплик совпадает с ground truth.", ""])
        for turn in case.scenario.turns:
            qa_lines.extend(
                [
                    f"Реплика {turn.index + 1}: {turn.utterance}",
                    "",
                    f"Ground truth: `{canonical_json(turn.spoken.model_dump(mode='json'))}`",
                    "",
                ]
            )
    qa_path.write_text("\n".join(qa_lines), encoding="utf-8")
    manifest_data = {
        "schema_version": 2,
        "dataset_version": DATASET_VERSION,
        "split": split,
        "origin": "synthetic_code",
        "blind_claim": False,
        "cases_file": cases_path.name,
        "qa_sample_file": qa_path.name,
        "cases_sha256": sha256_bytes(payload),
        "case_count": len(cases),
        "unique_users": len({case.scenario.user_id for case in cases}),
        "catalog_seed": SEEDS[split]["catalog"],
        "catalog_sha256": catalog_sha256(SEEDS[split]["catalog"]),
        "profile_seed": SEEDS[split]["profiles"],
        "scenario_seed": SEEDS[split]["scenarios"],
        "by_family": dict(sorted(counts.items())),
        "surface_families": sorted({case.surface_family_id for case in cases}),
        "normalized_dialogues_sha256": sha256_value(sorted(normalized_dialogue(case) for case in cases)),
        "statistical_target": {
            **statistical_target(),
            "achieved_independent_profiles": len(cases),
            "target_met": len(cases) >= 1600 and min(counts.values()) >= 100,
            "language_independence_proven": False,
        },
        "limitations": [
            "Synthetic code-generated Russian; neither human-authored nor real traffic.",
            "Public dev/validation are challenge sets, not blind holdouts.",
            "Language-family isolation is checked mechanically and does not prove linguistic independence.",
            "Canonical slot clauses are shared across public splits; validation is an exploratory challenge set.",
            "Cold-start scenarios only; profile histories are not injected into DemoProvider.",
        ],
    }
    manifest_data["manifest_sha256"] = sha256_value(manifest_data)
    manifest = V2Manifest.model_validate(manifest_data)
    manifest_path = output_dir / f"{DATASET_VERSION}-{split}.manifest.json"
    manifest_path.write_text(json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return cases_path, manifest_path


def load_manifest(manifest_path: Path, *, allow_final_holdout: bool = False) -> V2Manifest:
    """Validate manifest metadata without opening the case file."""
    raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = V2Manifest.model_validate(raw_manifest)
    if sha256_value(_manifest_payload(manifest)) != manifest.manifest_sha256:
        raise ValueError("manifest hash mismatch")
    if manifest.split == "final_holdout":
        if not allow_final_holdout:
            raise PermissionError("final_holdout requires an explicit consumption guard")
        try:
            manifest_path.resolve().relative_to(ROOT.resolve())
        except ValueError:
            pass
        else:
            raise ValueError("final_holdout must be supplied outside the repository")
        if manifest.origin != "independent_external":
            raise ValueError("final_holdout origin must be independent_external")
    return manifest


def load_cases(manifest_path: Path, manifest: V2Manifest) -> list[V2Case]:
    """Load frozen cases after the caller has applied any consumption policy."""
    cases_path = manifest_path.parent / manifest.cases_file
    payload = cases_path.read_bytes()
    if sha256_bytes(payload) != manifest.cases_sha256:
        raise ValueError("cases hash mismatch")
    cases = [V2Case.model_validate_json(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
    validate_cases(cases, expected_split=manifest.split)
    if len(cases) != manifest.case_count or len({case.scenario.user_id for case in cases}) != manifest.unique_users:
        raise ValueError("manifest counts disagree with cases")
    if Counter(case.scenario_family for case in cases) != Counter(manifest.by_family):
        raise ValueError("manifest family counts disagree with cases")
    if sha256_value(sorted(normalized_dialogue(case) for case in cases)) != manifest.normalized_dialogues_sha256:
        raise ValueError("normalized dialogue hash mismatch")
    return cases


def load_dataset(manifest_path: Path, *, allow_final_holdout: bool = False) -> tuple[V2Manifest, list[V2Case]]:
    manifest = load_manifest(manifest_path, allow_final_holdout=allow_final_holdout)
    if manifest.split == "final_holdout":
        raise PermissionError("final_holdout case bytes are loader-only after the CLI records consumption")
    return manifest, load_cases(manifest_path, manifest)


def _build_command(args: argparse.Namespace) -> None:
    if args.split == "final_holdout":
        raise SystemExit("final_holdout is loader-only and cannot be generated by this repository")
    cases, summary = generate_public_dataset(args.split, per_family=args.per_family)
    cases_path, manifest_path = write_dataset(cases, args.output_dir, split=args.split)
    print(canonical_json({**summary, "cases": str(cases_path), "manifest": str(manifest_path)}))


def _check_command(args: argparse.Namespace) -> None:
    left_manifest, left = load_dataset(args.left)
    right_manifest, right = load_dataset(args.right)
    result = validate_split_isolation(left, right, near_threshold=args.near_threshold)
    print(canonical_json({"left": left_manifest.split, "right": right_manifest.split, **result}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(required=True)
    build = subparsers.add_parser("build", help="build a public synthetic split")
    build.add_argument("--split", choices=(*PUBLIC_SPLITS, "final_holdout"), required=True)
    build.add_argument("--per-family", type=int, default=25)
    build.add_argument("--output-dir", type=Path, default=Path("artifacts/eval-v2-20260914"))
    build.set_defaults(func=_build_command)
    check = subparsers.add_parser("check-splits", help="check two frozen public manifests")
    check.add_argument("left", type=Path)
    check.add_argument("right", type=Path)
    check.add_argument("--near-threshold", type=float, default=0.88)
    check.set_defaults(func=_check_command)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
