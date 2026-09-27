"""Check that an LLM value agrees with the cited phrase and customer schema.

Declared aliases and their configured inflections can support a value. A phrase
naming a different value is a contradiction; unknown or ambiguous evidence stays
unresolved. LanguageProfile supplies the language rules. This module does not
read catalogs or evaluation labels.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal, Protocol

ResolutionStatus = Literal["canonical", "unresolved", "ambiguous", "unsupported"]

QUOTE_PAIRS = ("«»", '""', "“”", "‘’", "''")

#: Punctuation that is never part of a comparison key.
CANONICAL_TRIM = " \t\n«»\"“”‘’'.,;:!?"


def comparison_key(value: str) -> str:
    """Use the same normalized key for aliases, citations and resolver lookups."""
    return value.casefold().replace("ё", "е").strip(CANONICAL_TRIM)


def strip_enclosing_quotes(value: str) -> str:
    """Remove one layer of enclosing quotes, preserving case and inner punctuation.

    A cited title is evidence about a catalog object, and catalog titles do not
    carry the quoting the user typed. Case and ё are preserved because this is a
    display/lookup value, not a comparison key.
    """
    text = value.strip()
    for open_q, close_q in QUOTE_PAIRS:
        if len(text) >= 2 and text.startswith(open_q[0]) and text.endswith(close_q[-1]):
            return text[1:-1].strip()
    return text


#: Shortest stem that may prove two words are forms of the same lexeme. Below
#: this, a shared prefix is coincidence rather than morphology.
DEFAULT_MIN_STEM = 4

#: Longest inflectional ending tolerated between two forms of one lexeme.
DEFAULT_ENDING_TOLERANCE = 3


@dataclass(frozen=True)
class FieldSpec:
    """What a resolver may know about one schema field.

    Populated only from the adapter's injected schema/alias tables, never from
    ground truth: a resolver that could see expected values would be untestable
    and would silently overfit the development cohort.
    """

    name: str
    #: canonical_text(surface form) -> canonical value, from the schema enum
    enum: dict[str, object] = field(default_factory=dict)
    #: canonical_text(alias) -> canonical value, from the domain descriptor
    aliases: dict[str, object] = field(default_factory=dict)
    #: canonical_text(alias) -> scalar value, for bool/int fields
    scalar_aliases: dict[str, object] = field(default_factory=dict)
    #: unit text -> multiplier, for int fields
    numeric_units: dict[str, int] = field(default_factory=dict)

    def surfaces(self, canonical: object) -> set[str]:
        """Every declared surface form that denotes ``canonical``."""
        forms = {surface for surface, value in self.enum.items() if value == canonical}
        forms |= {alias for alias, value in self.aliases.items() if value == canonical}
        forms.discard("")
        return forms


@dataclass(frozen=True)
class Resolution:
    status: ResolutionStatus
    value: object = None
    #: What the adapter should cite in the user-facing issue.
    evidence: str = ""
    reason: str = ""


@dataclass(frozen=True)
class LanguageProfile:
    """Injectable morphological tolerance.

    A domain supplies this instead of the core carrying suffix lists: Russian
    inflection and an agglutinative language need different numbers, and a
    language with no inflection can set ``ending_tolerance=0`` and get exact
    surface matching only.
    """

    min_stem: int = DEFAULT_MIN_STEM
    ending_tolerance: int = DEFAULT_ENDING_TOLERANCE
    separators: tuple[str, ...] = (" ", "-", "–", "—", "/", "\t", "\n")
    negation_tokens: tuple[str, ...] = ()
    numeric_words: tuple[tuple[str, int], ...] = ()

    def tokens(self, text: str) -> list[str]:
        words = [text]
        for separator in self.separators:
            words = [part for word in words for part in word.split(separator)]
        return [word for word in (w.strip(" \t\n.,;:!?()[]«»\"'") for w in words) if word]

    def same_lexeme(self, a: str, b: str) -> bool:
        """True when ``a`` and ``b`` look like inflected forms of one lexeme.

        Deliberately conservative: a containment relation only counts when the
        shorter side is a real stem, and otherwise both sides must share a long
        prefix and differ by a short ending. This is what keeps "middle" from
        ever looking like a form of "начальный".
        """
        if not a or not b:
            return False
        if a == b:
            return True
        shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
        if shorter in longer:
            return longer.startswith(shorter) and len(shorter) >= self.min_stem and len(longer) - len(shorter) <= self.ending_tolerance
        shared = 0
        for x, y in zip(a, b, strict=False):
            if x != y:
                break
            shared += 1
        return shared >= self.min_stem and len(a) - shared <= self.ending_tolerance and len(b) - shared <= self.ending_tolerance

    def matches_surface(self, text: str, surface: str) -> bool:
        """True when ``text`` is a form of the declared ``surface``.

        Multi-word surfaces are covered token by token: "по машинному обучению"
        denotes the declared surface "машинное обучение" because every word of
        the surface has an inflected counterpart in the citation. This is why a
        domain can declare a phrase as a value surface without the core needing
        to know the language's case system.
        """
        text_tokens = self.tokens(text)
        surface_tokens = self.tokens(surface)
        if not text_tokens or not surface_tokens:
            return False
        return all(any(self.same_lexeme(token, word) for token in text_tokens) for word in surface_tokens)

    def has_negation(self, text: str) -> bool:
        return bool(set(self.tokens(text)) & set(self.negation_tokens))


class SemanticResolver(Protocol):
    """Decide whether cited evidence supports a proposed canonical value."""

    def resolve(self, spec: FieldSpec, value: object, source_text: str) -> Resolution: ...


class ExactSurfaceResolver:
    """Literal alias/enum matching: the historical, strictest behaviour."""

    def resolve(self, spec: FieldSpec, value: object, source_text: str) -> Resolution:
        permitted = dict(spec.enum)
        permitted.update(spec.aliases)
        raw = comparison_key(source_text)
        if raw not in permitted:
            return Resolution("unsupported", evidence=source_text, reason="source_not_declared")
        if not isinstance(value, str) or comparison_key(value) not in permitted:
            return Resolution("unsupported", evidence=str(value), reason="value_not_declared")
        if permitted[comparison_key(value)] != permitted[raw]:
            return Resolution("ambiguous", evidence=source_text, reason="value_evidence_mismatch")
        return Resolution("canonical", value=permitted[raw], evidence=source_text, reason="exact")


class InflectionalResolver:
    """Exact surface first, then inflectional coherence with declared surfaces.

    This resolver never interprets meaning. It accepts a canonical value when the
    citation is a form of a surface declared *for that value*, and it refuses
    when the citation is a form of a surface declared for a different value or
    for no value at all — which is exactly the unknown-vocabulary case that must
    stay unresolved rather than be snapped to the nearest enum member.
    """

    def __init__(self, profile: LanguageProfile | None = None):
        self.profile = profile or LanguageProfile()

    def _candidates(self, spec: FieldSpec, raw: str) -> set[object]:
        matched: set[object] = set()
        for surface, canonical in spec.enum.items():
            if self.profile.matches_surface(raw, surface):
                matched.add(canonical)
        for alias, canonical in spec.aliases.items():
            if self.profile.matches_surface(raw, alias):
                matched.add(canonical)
        return matched

    def resolve(self, spec: FieldSpec, value: object, source_text: str) -> Resolution:
        permitted = dict(spec.enum)
        permitted.update(spec.aliases)
        raw = comparison_key(source_text)

        # A literal declaration is always the strongest evidence, and a literal
        # contradiction always wins: this is what keeps a hallucinated canonical
        # value from being laundered by an unrelated citation.
        if raw in permitted:
            declared = permitted[raw]
            if value is None:
                return Resolution("canonical", value=declared, evidence=source_text, reason="null_value_resolved")
            if isinstance(value, str):
                value_key = comparison_key(value)
                if value_key in permitted and permitted[value_key] == declared:
                    return Resolution("canonical", value=declared, evidence=source_text, reason="exact")
            return Resolution("ambiguous", evidence=source_text, reason="value_evidence_mismatch")

        matched = self._candidates(spec, raw)
        if not matched:
            return Resolution("unsupported", evidence=source_text, reason="no_declared_surface")
        if len(matched) > 1:
            return Resolution("ambiguous", evidence=source_text, reason="multiple_declared_surfaces")
        canonical = next(iter(matched))

        if value is None:
            # Nothing was proposed, but the citation denotes exactly one declared
            # value: resolving it is reading the schema, not guessing.
            return Resolution("canonical", value=canonical, evidence=source_text, reason="null_value_resolved")
        if isinstance(value, str):
            value_key = comparison_key(value)
            if permitted.get(value_key) == canonical:
                return Resolution("canonical", value=canonical, evidence=source_text, reason="inflectional")
        return Resolution("ambiguous", evidence=source_text, reason="value_evidence_mismatch")


def resolve_scalar(spec: FieldSpec, value: object, source_text: str, profile: LanguageProfile | None = None) -> Resolution:
    """Scalar (bool/int) evidence coherence.

    Kept as a function rather than a class because the guarantees here are the
    ones that must not be relaxed by a pluggable resolver: an alias must state
    the same polarity/value as the proposal, and a number must literally appear
    in the citation (optionally through a declared unit).
    """
    raw = comparison_key(source_text)
    aliases = {comparison_key(alias): target for alias, target in spec.scalar_aliases.items()}
    if raw in aliases:
        target = aliases[raw]
        if type(target) is type(value) and target == value:
            return Resolution("canonical", value=value, evidence=source_text, reason="exact_scalar")
        return Resolution("ambiguous", evidence=source_text, reason="scalar_value_mismatch")

    if isinstance(value, bool):
        if profile is None:
            return Resolution("unsupported", evidence=source_text, reason="scalar_not_declared")
        # Polarity safety: the citation must denote declared aliases that all
        # agree with the proposed boolean. "практика" → False stays unsupported
        # even though "практика" is a declared alias, because that alias says True.
        matched = {target for alias, target in aliases.items() if type(target) is bool and profile.matches_surface(raw, alias)}
        if matched == {value}:
            return Resolution("canonical", value=value, evidence=source_text, reason="inflectional_scalar")
        return Resolution("unsupported", evidence=source_text, reason="scalar_not_declared")

    if type(value) is int:
        if raw in aliases and type(aliases[raw]) is int and aliases[raw] == value:
            return Resolution("canonical", value=value, evidence=source_text, reason="exact_scalar")
        # Decimal fragments are not evidence for an integer. In particular,
        # ``1.5`` must not yield the unrelated candidate ``5``.
        numbers = [int(token) for token in re.findall(r"(?<![\d.,])\d+(?![\d.,])", raw)]
        units = {
            multiplier
            for unit, multiplier in spec.numeric_units.items()
            if (profile.matches_surface(raw, unit) if profile is not None else re.search(r"(?<!\w)" + re.escape(unit) + r"(?!\w)", raw))
        }
        if numbers and units:
            interpreted = {number * multiplier for number in numbers for multiplier in units}
            if interpreted == {value}:
                return Resolution("canonical", value=value, evidence=source_text, reason="numeric_unit")
            return Resolution("ambiguous" if len(interpreted) > 1 else "unsupported", evidence=source_text, reason="numeric_unit_mismatch")
        if value in numbers:
            return Resolution("canonical", value=value, evidence=source_text, reason="numeric_literal")
        if profile is not None and spec.numeric_units:
            tokens = {comparison_key(token) for token in profile.tokens(raw)}
            word_numbers = {number for word, number in profile.numeric_words if comparison_key(word) in tokens}
            if len(word_numbers) > 1:
                return Resolution("ambiguous", evidence=source_text, reason="multiple_numeric_words")
            if word_numbers:
                number = next(iter(word_numbers))
                for unit, multiplier in spec.numeric_units.items():
                    if profile.matches_surface(raw, unit) and number * multiplier == value:
                        return Resolution("canonical", value=value, evidence=source_text, reason="numeric_word_unit")
        return Resolution("unsupported", evidence=source_text, reason="numeric_mismatch")

    if isinstance(value, str):
        # Free text (titles, names): the citation must equal the value, and the
        # value is stored without the user's enclosing quotes so catalog lookup
        # and downstream comparison see the title as declared.
        normalized = strip_enclosing_quotes(value)
        if comparison_key(normalized) == raw:
            return Resolution("canonical", value=normalized, evidence=source_text, reason="exact_text")
        return Resolution("unsupported", evidence=source_text, reason="text_mismatch")
    return Resolution("unsupported", evidence=source_text, reason="unsupported_type")
