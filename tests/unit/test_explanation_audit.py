from evals.explanations import audit_text

from recagent.catalog import generate_catalog


def test_false_facts_are_rejected_even_without_evidence():
    item = generate_catalog()[0]
    result = audit_text(f"Жанр: {item.genre}. Сезонов: 99. Это лучший фильм года.", item)
    assert result["claims"] == 3
    assert len(result["invalid"]) == 2


def test_history_claim_requires_a_real_source():
    item = generate_catalog()[0]
    assert audit_text("Этот жанр есть в вашей истории.", item)["invalid"]
    assert not audit_text("Этот жанр есть в вашей истории.", item, [item])["invalid"]
