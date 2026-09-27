"""Freeze a small AI-authored Russian operation contract; never reads agent outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 137
DATASET = "product-contract-ru-v1"


def turn(message, spoken, *, intent="discovery", state="recommend", policy="not_applicable", **extra):
    return {
        "user": message,
        "expected_state": state,
        "recommendation_exists": state == "recommend",
        "spoken": spoken,
        "expected_query": {"intent": intent},
        "commit_policy": policy,
        **extra,
    }


def build_cohort():
    """Annotations are authored with messages, not inferred from runtime Query."""
    title = "Тишина в ожидании поезда"
    cases = [
        (
            "explicit-course",
            [
                turn(
                    "Для занятий выбираю курс: тема Python, уровень начальный, обязательно практика.",
                    {"kind": "course", "genre": "python", "level": "начальный", "practical": True},
                )
            ],
        ),
        (
            "numeric-correction",
            [
                turn(
                    "Подберите сериал: жанр фантастика, серии максимум 35 минут.",
                    {"kind": "series", "genre": "фантастика", "max_minutes": 35},
                ),
                turn(
                    "Исправлю длительность: максимум 55 минут, остальное сохраняем.",
                    {"kind": "series", "genre": "фантастика", "max_minutes": 55},
                ),
            ],
        ),
        (
            "accumulate-exclusions",
            [
                turn("Нужен фильм нейтрального тона, без драмы.", {"kind": "film", "tone": "нейтральный", "excluded_genres": ["драма"]}),
                turn("И комедию тоже исключите.", {"kind": "film", "tone": "нейтральный", "excluded_genres": ["драма", "комедия"]}),
            ],
        ),
        (
            "include-exact",
            [
                turn(
                    "Ищу фильм: тон нейтральный, исключить драму и комедию.",
                    {"kind": "film", "tone": "нейтральный", "excluded_genres": ["драма", "комедия"]},
                ),
                turn(
                    "Комедию снова разрешаю; запрет на драму остаётся.",
                    {"kind": "film", "tone": "нейтральный", "excluded_genres": ["драма"]},
                ),
            ],
        ),
        (
            "clear-limit",
            [
                turn("Фильм жанра приключения, не длиннее 110 минут.", {"kind": "film", "genre": "приключения", "max_minutes": 110}),
                turn("Снимите ограничение длительности; приключения оставьте.", {"kind": "film", "genre": "приключения"}),
            ],
        ),
        (
            "domain-change",
            [
                turn("Хочется фильм-драму длительностью до 100 минут.", {"kind": "film", "genre": "драма", "max_minutes": 100}),
                turn(
                    "Начнём другой подбор: курс по Python для начинающего, без практики.",
                    {"kind": "course", "genre": "python", "level": "начальный", "practical": False},
                ),
            ],
        ),
        (
            "navigation",
            [
                turn(
                    f"Откройте единственный объект с точным названием «{title}».",
                    {"named_title": title},
                    intent="navigation",
                    exact_recommendation_ids=["it-00001"],
                )
            ],
        ),
        (
            "similar",
            [
                turn(
                    f"Порекомендуйте другой фильм, похожий на «{title}».",
                    {"kind": "film", "seed_title": title},
                    intent="similar",
                    forbidden_recommendation_ids=["it-00001"],
                )
            ],
        ),
        ("mood", [turn("Для лёгкого настроения выберу сериал с лёгким тоном.", {"kind": "series", "tone": "лёгкий"}, intent="mood")]),
        (
            "negative-tone-more",
            [
                turn("Предложите фильм-детектив, только не мрачный.", {"kind": "film", "genre": "детектив", "excluded_tones": ["мрачный"]}),
                turn(
                    "Покажите ещё, сохранив условия.",
                    {"kind": "film", "genre": "детектив", "excluded_tones": ["мрачный"]},
                    disjoint_from_turns=[0],
                ),
            ],
        ),
        (
            "unknown-level-resolution",
            [
                turn(
                    "Подбираю курс Python, уровень грандмастер.",
                    {"kind": "course", "genre": "python"},
                    state="clarify",
                    policy="required_unresolved",
                ),
                turn("Уточняю уровень: начальный.", {"kind": "course", "genre": "python", "level": "начальный"}),
            ],
        ),
        (
            "optional-preference",
            [
                {
                    "user": "В этот раз выбираю фильм.",
                    "spoken": {"kind": "film"},
                    "expected_query": {"intent": "discovery"},
                    "allowed_states": ["clarify", "recommend"],
                    "presence_by_state": {"clarify": False, "recommend": True},
                    "commit_policy": "optional_after_commit",
                },
                turn("Жанр не важен; тон нужен нейтральный.", {"kind": "film", "tone": "нейтральный"}),
            ],
        ),
    ]
    dialogues = []
    for key, turns in cases:
        # Full literal turn spans are evidence provenance, not a parser or a
        # claim that every word supports every field. Spoken values above are
        # the author's independent cumulative annotations.
        for index, entry in enumerate(turns):
            entry["source_annotations"] = {
                "author": "AI assistant",
                "kind": "explicit_synthetic_annotation",
                "evidence": [{"turn": prior, "quote": turns[prior]["user"]} for prior in range(index + 1)],
                "scope": "Cumulative explicitly stated constraints; corrections override only their addressed condition.",
            }
        dialogues.append({"id": key, "tags": [key], "turns": turns})
    return {
        "schema_version": 1,
        "dataset_id": DATASET,
        "version": "1.0.0",
        "split": "contract",
        "origin": "ai_authored_synthetic",
        "status": "frozen_before_candidate_outputs",
        "generation_seed": SEED,
        "catalog": {"seed": SEED, "known_titles": {"navigation": {"id": "it-00001", "title": title, "kind": "film"}}},
        "model": {"name": "qwen3:8b", "expected_digest": "500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41"},
        "configuration": {"mode": "ollama", "question_policy": "adaptive", "max_questions": 3, "history": "cold_start"},
        "limitations": [
            "AI-authored synthetic diagnostic cases, not human annotation or independent final holdout.",
            "Authored after inspecting older DEV failure classes; independence from all prior development is not claimed.",
            "Frozen before outputs on this cohort; fixed order, seed controls the public synthetic catalog only.",
            "No preference-utility score. Similar only checks explicit spoken constraints and exclusion of the named seed.",
            "expected_query is diagnostic only; observable acceptance cannot derive labels from runtime state.",
            "Clear/include/domain-change existence checks do not prove recall; component invariants test state transitions separately.",
        ],
        "dialogues": dialogues,
    }


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/product_contract_ru_v1.json")
    args = parser.parse_args()
    payload = encoded(build_cohort())
    manifest = {
        "dataset_id": DATASET,
        "version": "1.0.0",
        "origin": "ai_authored_synthetic",
        "seed": SEED,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dialogs": len(build_cohort()["dialogues"]),
        "turns": sum(len(d["turns"]) for d in build_cohort()["dialogues"]),
        "signature": {
            "scheme": "sha256_content_address",
            "cryptographic_author_signature": None,
            "note": "Integrity receipt, not proof of human authorship or a private-key signature.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    args.output.with_suffix(".manifest.json").write_bytes(encoded(manifest))
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
