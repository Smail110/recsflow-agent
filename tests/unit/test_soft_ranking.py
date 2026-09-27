"""Ranking controls use invented catalogs, independent of dialogue DEV labels."""

from copy import deepcopy

import pytest

from recagent.domains.base import FieldSpec
from recagent.domains.demo import domain_spec
from recagent.models import Item
from recagent.ranking import soft_rank


def item(item_id, *, genre=None, tone=None, quality=0.5, **values):
    return Item(
        id=item_id,
        title=item_id,
        kind="film",
        genre=genre,
        tone=tone,
        quality=quality,
        description="",
        **values,
    )


def rank(items, **kwargs):
    return soft_rank(items, fields=domain_spec().fields, intent=kwargs.pop("intent", "discovery"), **kwargs)


def ids(items):
    return [entry.id if isinstance(entry, Item) else entry["id"] for entry in items]


def test_similar_seed_changes_top_result_over_provider_quality():
    dramatic = item("dramatic", genre="драма", tone="мрачный", quality=0.2)
    comic = item("comic", genre="комедия", tone="лёгкий", quality=0.9)
    candidates = [comic, dramatic]
    assert ids(rank(candidates, intent="similar", seed=item("seed", genre="драма", tone="мрачный")))[0] == "dramatic"
    assert ids(rank(candidates, intent="similar", seed=item("seed", genre="комедия", tone="лёгкий")))[0] == "comic"


def test_verified_seed_similarity_precedes_speculative_genre_hint():
    similar = item("similar", genre="драма", tone="мрачный", quality=0.1)
    speculative = item("speculative", genre="фантастика", tone="лёгкий", quality=0.9)
    result = rank(
        [speculative, similar], intent="similar", seed=item("seed", genre="драма", tone="мрачный"),
        soft_preferences=[("genre", "фантастика")], quality_field="quality",
    )
    assert ids(result)[0] == "similar"


def test_current_descriptive_wish_precedes_old_history_for_discovery():
    comedy = item("comedy", genre="комедия", quality=0.2)
    drama = item("drama", genre="драма", quality=0.9)
    result = rank(
        [drama, comedy], history=[item("watched", genre="драма")],
        soft_preferences=[("genre", "комедия")], quality_field="quality",
    )
    assert ids(result)[0] == "comedy"


def test_navigation_ignores_speculative_genre_hint():
    exact = item("exact", genre="драма", quality=0.1)
    speculative = item("speculative", genre="фантастика", quality=0.9)
    assert ids(rank([exact, speculative], intent="navigation", soft_preferences=[("genre", "фантастика")])) == [
        "exact", "speculative"
    ]


def test_history_affinity_and_explicit_like_affect_discovery():
    science = item("science", genre="фантастика")
    adventure = item("adventure", genre="приключения")
    history = [item("watched", genre="приключения"), item("liked", genre="фантастика")]
    assert ids(rank([science, adventure], history=history[:1])) == ["adventure", "science"]
    assert ids(rank([adventure, science], history=history, liked_ids={"liked"})) == ["science", "adventure"]


def test_hard_filtered_candidates_are_never_replaced_by_seed_or_history():
    allowed = item("allowed", genre="драма")
    outside = item("excluded", genre="комедия")
    result = rank([allowed], intent="similar", seed=outside, history=[outside], liked_ids={outside.id})
    assert result == [allowed]
    assert result[0] is allowed


def test_navigation_preserves_verified_retrieval_order_despite_history():
    exact = item("exact", genre="драма")
    distractor = item("distractor", genre="комедия", quality=1)
    history = [item("liked", genre="комедия")]
    assert ids(rank([exact, distractor], intent="navigation", history=history, liked_ids={"liked"})) == ["exact", "distractor"]


def test_unknown_seed_attributes_do_not_reward_unknown_candidates():
    unknown = item("unknown")
    known = item("known", genre="драма")
    assert ids(rank([known, unknown], intent="similar", seed=item("seed"))) == ["known", "unknown"]


def test_sparse_candidate_does_not_get_perfect_similarity_by_omission():
    partial = item("partial", genre="драма")
    full = item("full", genre="драма", tone="мрачный")
    seed = item("seed", genre="драма", tone="мрачный")
    assert ids(rank([partial, full], intent="similar", seed=seed)) == ["full", "partial"]


def test_unrelated_domain_projection_multi_values_and_false_boolean_are_supported():
    fields = (
        FieldSpec(name="segment", item_field="family", value_type="enum"),
        FieldSpec(name="tags", value_type="enum", cardinality="multi"),
        FieldSpec(name="wireless", value_type="boolean"),
    )
    seed = {"id": "reference", "family": "input", "tags": ["silent", "compact"], "wireless": False}
    far = {"id": "far", "family": "display", "tags": ["bright"], "wireless": True}
    near = {"id": "near", "family": "input", "tags": ["compact"], "wireless": False}
    unknown = {"id": "unknown", "family": "input", "tags": ["compact"]}
    result = soft_rank([far, unknown, near], fields=fields, intent="similar", seed=seed)
    assert ids(result) == ["near", "unknown", "far"]


def test_numerical_constraints_are_not_assumed_to_be_similarity_dimensions():
    seed = {"id": "seed", "price": 100, "category": "accessory"}
    cheaper = {"id": "cheaper", "price": 50, "category": "accessory"}
    same_price = {"id": "same-price", "price": 100, "category": "accessory"}
    fields = (FieldSpec(name="price", value_type="integer"), FieldSpec(name="category", value_type="enum"))
    assert ids(soft_rank([cheaper, same_price], fields=fields, intent="similar", seed=seed)) == ["cheaper", "same-price"]


def test_no_personalization_preserves_relevance_and_does_not_mutate_inputs():
    candidates = [item("z", quality=0.1), item("a", quality=1)]
    before = deepcopy(candidates)
    assert ids(rank(candidates)) == ["z", "a"]
    assert ids(rank(candidates, order=["a", "z"], limit=1)) == ["a"]
    assert candidates == before
    assert rank(candidates, limit=0) == []


def test_equal_preferences_keep_provider_relevance_and_history_duplicates_do_not_reweight_it():
    candidates = [item("first", genre="драма"), item("second", genre="комедия")]
    drama = item("old-drama", genre="драма")
    comedy = item("old-comedy", genre="комедия")
    assert ids(rank(candidates, history=[drama, comedy, comedy])) == ["first", "second"]


def test_public_quality_prior_is_opt_in_and_accepts_a_domain_declared_field():
    candidates = [{"id": "first", "public_score": -8}, {"id": "better", "public_score": -2}]
    before = deepcopy(candidates)
    options = {"fields": (), "intent": "discovery"}
    assert ids(soft_rank(candidates, **options)) == ["first", "better"]
    assert ids(soft_rank(candidates, **options, quality_field="public_score")) == ["better", "first"]
    assert candidates == before


@pytest.mark.parametrize("preference", ["seed", "like", "history"])
def test_affinity_precedes_public_quality(preference):
    fields = (FieldSpec(name="category", value_type="enum"),)
    relevant = {"id": "relevant", "category": "input", "rating": 1}
    unrelated = {"id": "unrelated", "category": "display", "rating": 9}
    reference = {"id": "reference", "category": "input"}
    options = {"intent": "discovery", "history": [reference]}
    if preference == "seed":
        options = {"intent": "similar", "seed": reference}
    elif preference == "like":
        options["history"] = [reference, {"id": "other", "category": "display"}]
        options["liked_ids"] = ["reference"]
    assert ids(soft_rank([unrelated, relevant], fields=fields, quality_field="rating", **options)) == ["relevant", "unrelated"]


def test_navigation_ignores_quality_and_affinity_and_keeps_unlisted_items_stable():
    candidates = [{"id": "z", "category": "input", "rating": 1}, {"id": "a", "category": "display", "rating": 9}]
    options = {
        "fields": (FieldSpec(name="category", value_type="enum"),),
        "intent": "navigation",
        "quality_field": "rating",
        "history": [candidates[1]],
    }
    assert ids(soft_rank(candidates, **options)) == ["z", "a"]
    assert ids(soft_rank(candidates, order=["a", "z"], **options)) == ["a", "z"]
    assert ids(soft_rank(candidates, order=[], **options)) == ["z", "a"]


@pytest.mark.parametrize("bad", [None, "9", True, float("nan"), float("inf"), -float("inf"), 1 + 2j, 10**1000])
def test_incomplete_or_invalid_quality_disables_prior_for_entire_pool(bad):
    candidates = [{"id": "low", "quality": -5}, {"id": "high", "quality": 5}, {"id": "unknown", "quality": bad}]
    options = {"fields": (), "intent": "discovery", "quality_field": "quality"}
    assert ids(soft_rank(candidates, **options)) == ["low", "high", "unknown"]
    assert ids(soft_rank(candidates, order=["unknown", "low", "high"], **options)) == ["unknown", "low", "high"]


def test_missing_quality_does_not_zero_impute_or_disable_valid_affinity():
    candidates = [
        {"id": "first", "category": "display", "rating": -5},
        {"id": "better", "category": "display", "rating": 5},
        {"id": "unknown", "category": "input"},
    ]
    options = {"fields": (FieldSpec(name="category", value_type="enum"),), "intent": "similar", "quality_field": "rating"}
    assert ids(soft_rank(candidates, **options)) == ["first", "better", "unknown"]
    assert ids(soft_rank(candidates, seed={"category": "input"}, **options)) == ["unknown", "first", "better"]


def test_equal_quality_is_stable_and_prior_never_adds_filtered_items():
    eligible = [{"id": "z", "rating": 7}, {"id": "a", "rating": 7}]
    excluded = {"id": "excluded", "rating": 100}
    options = {"fields": (), "intent": "similar", "quality_field": "rating", "seed": excluded, "history": [excluded]}
    assert ids(soft_rank(eligible, **options)) == ["z", "a"]
    assert ids(soft_rank(eligible, order=["a", "z"], **options)) == ["a", "z"]
    result = soft_rank(eligible, limit=1, **options)
    assert result == eligible[:1] and result[0] is eligible[0]
    assert soft_rank(eligible, limit=0, **options) == []
