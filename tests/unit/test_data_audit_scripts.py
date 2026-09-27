from types import SimpleNamespace

import pytest
from scripts import audit_cross_source_leakage as cross_source
from scripts import audit_dataset_diversity as diversity


def test_diversity_uses_normalized_word_bigrams():
    assert diversity.word_bigrams("Ёж, идёт домой!") == {("еж", "идет"), ("идет", "домой")}
    assert diversity.bigram_jaccard(diversity.word_bigrams("один два три"), diversity.word_bigrams("один два четыре")) == pytest.approx(
        1 / 3
    )
    case = SimpleNamespace(scenario=SimpleNamespace(turns=[SimpleNamespace(utterance="Один два"), SimpleNamespace(utterance="Три четыре")]))
    assert diversity.dialogue_bigrams(case) == {
        ("один", "два"),
        ("два", "||"),
        ("||", "три"),
        ("три", "четыре"),
    }


def test_cross_source_uses_token_jaccard_and_separates_exact_matches():
    assert cross_source.token_jaccard("А ёж идёт домой", "а еж идет в сад") == pytest.approx(3 / 6)
    common = "один два три четыре пять шесть семь восемь девять"
    result = cross_source.compare_groups(
        [("left-near", f"{common} десять"), ("left-exact", "Точный ёж!")],
        [("right-near", f"{common} одиннадцать"), ("right-exact", "точный еж")],
    )
    assert result["exact_pairs"] == [["left-exact", "right-exact"]]
    assert result["near_pairs"] == [{"left": "left-near", "right": "right-near", "token_jaccard": 0.8182}]


def test_public_loader_rejects_final_before_case_bytes(monkeypatch, tmp_path):
    manifest = SimpleNamespace(split="final_holdout")
    monkeypatch.setattr(diversity, "load_manifest", lambda _path: manifest)

    def unexpected_case_read(*args, **kwargs):
        raise AssertionError("final case bytes must not be read")

    monkeypatch.setattr(diversity, "load_cases", unexpected_case_read)
    with pytest.raises(ValueError, match="only public dev or validation"):
        diversity.load_public_dataset(tmp_path / "final.manifest.json")


def test_cross_source_jsonl_loaders_require_their_text_fields(tmp_path):
    training = tmp_path / "training.jsonl"
    training.write_text('{"text":"Запрос"}\n', encoding="utf-8")
    payload, rows = cross_source.load_training_smoke(training)
    assert payload == training.read_bytes()
    assert rows == [("0", "Запрос")]

    external = tmp_path / "external.jsonl"
    external.write_text('{"id":"pair-1","text_1":"Первый","text_2":"Второй"}\n', encoding="utf-8")
    _, rows = cross_source.load_external_language(external)
    assert rows == [("pair-1:1", "Первый"), ("pair-1:2", "Второй")]

    external.write_text('{"id":"broken","text_1":"Первый"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="text_2 must be a non-empty string"):
        cross_source.load_external_language(external)
