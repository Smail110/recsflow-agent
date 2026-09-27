import pytest
from pydantic import ValidationError

from recagent.contracts import Constraint, SourceSpan, project_constraints_to_query
from recagent.domains.base import DomainSpec, FieldSpec, ProviderCapabilities


def span():
    return SourceSpan(turn_id="t1", start=0, end=5, text="фильм")


def test_contracts_are_frozen_and_strict():
    constraint = Constraint(id="c1", field="max_minutes", op="lte", value=90, turn_id="t1", domain_version="demo/1", source_spans=(span(),))
    with pytest.raises(ValidationError):
        Constraint.model_validate({**constraint.model_dump(), "op": "between"}, strict=True)
    with pytest.raises((ValidationError, TypeError)):
        constraint.value = 60


def test_projection_keeps_unsupported_constraints_as_residual():
    constraints = [
        Constraint(id="g", field="genre", op="eq", value="драма", turn_id="t1", domain_version="1"),
        Constraint(id="x", field="audience", op="eq", value="adult", turn_id="t1", domain_version="1"),
        Constraint(id="n", field="max_minutes", op="lte", value=90, turn_id="t1", domain_version="1"),
    ]
    query, residual = project_constraints_to_query(constraints)
    assert query.genre == "драма"
    assert [item.id for item in residual] == ["x", "n"]


def test_domain_and_capability_descriptors_are_serializable():
    spec = DomainSpec(id="demo", version="1", request_schema="recagent.models.Query", fields=(FieldSpec(name="genre", value_type="enum"),))
    caps = ProviderCapabilities(provider_id="memory", version="1", supports_lookup=True)
    assert spec.model_dump(mode="json")["fields"][0]["name"] == "genre"
    assert caps.supported_fields == ()
