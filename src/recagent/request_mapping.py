"""Validate structured updates against an injected customer schema."""

import re

from pydantic import BaseModel, ValidationError

from .context_nli import ContextNLI, EvidenceInput
from .interpretation import ConstraintUpdate, InterpretationIssue, StructuredRequest
from .resolution import FieldSpec, InflectionalResolver, LanguageProfile, SemanticResolver, comparison_key, resolve_scalar


def canonical_text(value: str) -> str:
    # Use the resolver's normalization on both sides of comparisons.
    return comparison_key(value)


class SchemaRequestAdapter:
    def __init__(
        self,
        model: type[BaseModel],
        *,
        aliases: dict[str, dict[str, str]] | None = None,
        exclusions: dict[str, str] | None = None,
        domain_field: str | None = None,
        field_labels: dict[str, str] | None = None,
        display_labels: dict[str, str] | None = None,
        field_questions: dict[str, str] | None = None,
        contextual_domain_questions: dict[str, str] | None = None,
        no_preference_markers: dict[str, tuple[str, ...]] | None = None,
        clear_cues: dict[str, tuple[str, ...]] | None = None,
        clear_markers: tuple[str, ...] = (),
        keep_markers: tuple[str, ...] = (),
        soft_description_fields: set[str] | None = None,
        soft_rank_fields: set[str] | None = None,
        soft_rank_aliases: dict[str, tuple[str, str]] | None = None,
        soft_request_markers: tuple[str, ...] = (),
        soft_descriptor_pattern: str | None = None,
        explicit_domain_surfaces: tuple[str, ...] = (),
        hard_preference_markers: tuple[str, ...] = (),
        exclusion_markers: tuple[str, ...] = (),
        inclusion_markers: tuple[str, ...] = (),
        negated_exclusion_markers: tuple[str, ...] = (),
        scalar_aliases: dict[str, dict[str, object]] | None = None,
        numeric_units: dict[str, dict[str, int]] | None = None,
        implied_values: dict[str, dict[object, dict[str, object]]] | None = None,
        constraint_operators: dict[str, str] | None = None,
        constraint_item_fields: dict[str, str] | None = None,
        canonical_exclusions: set[str] | None = None,
        resolver: SemanticResolver | None = None,
        language_profile: LanguageProfile | None = None,
        context_verifier: ContextNLI | None = None,
        preference_fields: set[str] | None = None,
    ):
        self.model = model
        self.aliases = aliases or {}
        self.exclusions = exclusions or {}
        self.domain_field = domain_field
        self.field_labels = field_labels or {}
        # UI copy is separate from the extraction descriptor: wording changes
        # must not silently alter the LLM prompt or accepted values.
        self.display_labels = display_labels or self.field_labels
        self.field_questions = field_questions or {}
        self.contextual_domain_questions = contextual_domain_questions or {}
        self.no_preference_markers = no_preference_markers or {}
        self.clear_cues = clear_cues or {}
        self.clear_markers = clear_markers
        self.keep_markers = keep_markers
        self.soft_description_fields = frozenset(soft_description_fields or ())
        self.soft_rank_fields = frozenset(soft_rank_fields or ())
        self.soft_rank_aliases = soft_rank_aliases or {}
        self.soft_request_markers = soft_request_markers
        self.soft_descriptor_pattern = soft_descriptor_pattern
        self.explicit_domain_surfaces = explicit_domain_surfaces
        self.hard_preference_markers = hard_preference_markers
        self.exclusion_markers = exclusion_markers
        self.inclusion_markers = inclusion_markers
        self.negated_exclusion_markers = negated_exclusion_markers
        self.scalar_aliases = scalar_aliases or {}
        self.numeric_units = numeric_units or {}
        # Declared schema relations may fill a value only when the source
        # constraint was already accepted.  They are not a second text parser.
        self.implied_values = implied_values or {}
        self.constraint_operators = constraint_operators or {}
        self.constraint_item_fields = constraint_item_fields or {}
        self.canonical_exclusions = frozenset(canonical_exclusions or ())
        # Morphological tolerance belongs to the domain, not to the core: a
        # language without inflection can pass ending_tolerance=0 and get exact
        # surface matching only. The same profile drives enum and scalar paths.
        self.language_profile = language_profile or LanguageProfile()
        # Accept inflections of declared vocabulary by default. A different
        # resolver can be injected without changing the adapter contract.
        self.resolver: SemanticResolver = resolver if resolver is not None else InflectionalResolver(self.language_profile)
        # Explicit opt-in for an experimental full-context enum verifier.
        # Default normalization remains the established resolver path.
        self.context_verifier = context_verifier
        # A schema-declared preference role is distinct from a field that can
        # also serve soft ranking. Callers opt in explicitly for claim wording.
        self.preference_fields = frozenset(preference_fields or ())
        self.enums = {}
        for name, schema in model.model_json_schema()["properties"].items():
            options = schema.get("anyOf", [schema])
            values = next((option["enum"] for option in options if "enum" in option), None)
            if values:
                self.enums[name] = {canonical_text(value): value for value in values}
        self._field_specs = {
            name: FieldSpec(
                name=name,
                # Defensive copies: FieldSpec is shared per adapter
                # instance and must not alias the descriptor tables.
                enum=dict(self.enums.get(name, {})),
                aliases={canonical_text(k): v for k, v in self.aliases.get(name, {}).items()},
                scalar_aliases={canonical_text(k): v for k, v in self.scalar_aliases.get(name, {}).items()},
                numeric_units={canonical_text(k): v for k, v in self.numeric_units.get(name, {}).items()},
            )
            for name in set(self.model.model_fields) | set(self.enums) | set(self.aliases)
        }

    def _field_spec(self, name: str) -> FieldSpec:
        return self._field_specs.get(name) or FieldSpec(name=name)

    @property
    def descriptor(self) -> dict:
        return {
            "schema": self.model.model_json_schema(),
            "aliases": self.aliases,
            "exclusion_fields": self.exclusions,
            "field_labels": self.field_labels,
            "scalar_aliases": self.scalar_aliases,
            "numeric_units": self.numeric_units,
            "implied_values": self.implied_values,
            "constraint_operators": self.constraint_operators,
            "constraint_item_fields": self.constraint_item_fields,
            "canonical_exclusions": sorted(self.canonical_exclusions),
        }

    def _scalar_matches(self, name: str, value: object, source: str) -> bool:
        """Historical scalar matcher used by attribution reports.

        Runtime normalization uses resolution.resolve_scalar instead.
        """
        raw = canonical_text(source)
        aliases = {canonical_text(k): v for k, v in self.scalar_aliases.get(name, {}).items()}
        if raw in aliases:
            return type(value) is type(aliases[raw]) and value == aliases[raw]
        if isinstance(value, str):
            return canonical_text(value) == raw
        if type(value) is int:
            parts = raw.split()
            try:
                number = int(parts[0])
            except (ValueError, IndexError):
                return False
            if len(parts) == 1:
                return number == value
            multiplier = self.numeric_units.get(name, {}).get(" ".join(parts[1:]))
            return multiplier is not None and number * multiplier == value
        return False

    def _issue(self, field: str, value: object, *, kind="unsupported_constraint", source_text: str | None = None) -> InterpretationIssue:
        options = ", ".join(self.enums.get(field, {}).values())
        label = self.display_labels.get(field, "условию запроса")
        message = f"Не могу однозначно применить «{value}» к полю «{label}»."
        message += f" Уточните значение: {options}." if options else " Уточните это условие."
        return InterpretationIssue(kind=kind, field=field, value=str(value), message=message, source_text=source_text)

    def _explicit_no_preference(self, field: str, source_text: str) -> bool:
        """Accept a declared opt-out only when the citation has no concrete value."""
        markers = self.no_preference_markers.get(field, ())
        source = canonical_text(source_text)
        matched = [marker for marker in markers if self.language_profile.matches_surface(source, marker)]
        if not matched:
            return False
        # Any negation outside a declared marker makes the opt-out ambiguous.
        # Removing the complete marker preserves valid phrases such as
        # "жанр не важно" and "без ограничений", but rejects both
        # "не любой жанр" and "любой жанр не подходит".
        outside_markers = source
        for marker in matched:
            outside_markers = re.sub(rf"\b{re.escape(canonical_text(marker))}\b", " ", outside_markers)
        if any(re.search(rf"\b{re.escape(negation)}\b", outside_markers) for negation in self.language_profile.negation_tokens):
            return False
        if self._clear_citation_conflicts(source):
            return False
        if field in self.clear_cues and self._clear_fields(source) != {field}:
            return False
        if field in self.enums:
            surfaces = (*self.enums[field], *self.aliases.get(field, {}))
            return not any(self.language_profile.matches_surface(source, surface) for surface in surfaces)
        if re.search(r"(?<![\d.,])\d+(?![\d.,])", source):
            return False
        tokens = {canonical_text(token) for token in self.language_profile.tokens(source)}
        return not any(canonical_text(word) in tokens for word, _ in self.language_profile.numeric_words)

    def _clear_fields(self, source_text: str) -> set[str]:
        source = canonical_text(source_text)
        return {
            field
            for field, cues in self.clear_cues.items()
            if any(self.language_profile.matches_surface(source, canonical_text(cue)) for cue in cues)
        }

    def _clear_citation_conflicts(self, source_text: str) -> bool:
        source = canonical_text(source_text)
        citation = source.strip(" .!?")
        return bool(re.search(r"[,;.!?]", citation)) or any(
            self.language_profile.matches_surface(source, canonical_text(marker)) for marker in self.keep_markers
        )

    def soft_preference(self, update, message: str) -> dict[str, object] | None:
        """Downgrade an unsupported descriptive facet, never an explicit filter.

        The model's guessed canonical value may guide ranking only for fields
        declared by this domain. It cannot enter active constraints or explain
        a catalog attribute that the provider has not verified.
        """
        field = update.field
        if (
            field not in self.soft_description_fields
            or update.operation != "set"
            or not isinstance(update.value, str)
            or update.value not in self.enums.get(field, {}).values()
            or not self._source_is_cited(update.source_text, message)
        ):
            return None
        checked = self.normalize(StructuredRequest(updates=[update]), message)
        if checked.updates or len(checked.issues) != 1 or checked.issues[0].kind != "unsupported_constraint":
            return None
        if not self._is_soft_description(field, update.source_text, message):
            return None
        rank_field, rank_value = self._soft_rank_target(update.source_text, field, update.value)
        return {"field": rank_field, "value": rank_value, "source_field": field, "source_text": update.source_text}

    def soft_issue(self, issue: InterpretationIssue, message: str) -> dict[str, object] | None:
        """Treat a cited, non-mandatory unknown description as an unranked wish."""
        source = issue.source_text
        if (
            issue.kind != "unsupported_constraint"
            or issue.field not in self.soft_description_fields
            or not source
            or not self._source_is_cited(source, message)
        ):
            return None
        if not self._is_soft_description(issue.field, source, message):
            return None
        rank_field, rank_value = self._soft_rank_target(source, issue.field, None)
        return {"field": rank_field, "value": rank_value, "source_field": issue.field, "source_text": source}

    def _soft_rank_target(self, source: str, field: str, value: str | None) -> tuple[str, str | None]:
        """Use only declared approximate aliases; conflicting cues stay unranked."""
        matches = {
            target
            for surface, target in self.soft_rank_aliases.items()
            if self.language_profile.matches_surface(canonical_text(source), canonical_text(surface))
            and target[0] in self.soft_rank_fields
            and target[1] in self.enums.get(target[0], {}).values()
        }
        if len(matches) == 1:
            return next(iter(matches))
        if matches:
            return field, None
        return field, value if field in self.soft_rank_fields else None

    def exact_domain_answer(self, message: str, question: str | None) -> ConstraintUpdate | None:
        """Resolve a one-word answer only among options in our own format question."""
        field = self.domain_field
        if not field or not question:
            return None
        answer = canonical_text(message)
        matches = {
            value
            for surface, value in self.aliases.get(field, {}).items()
            if canonical_text(surface) == answer
            and self.language_profile.matches_surface(canonical_text(question), canonical_text(surface))
        }
        if len(matches) != 1:
            return None
        return ConstraintUpdate(field=field, operation="set", value=next(iter(matches)), source_text=message.strip())

    def recover_explicit_domain(self, request: StructuredRequest, message: str) -> ConstraintUpdate | None:
        """Complete a missing format only from one declared, literal user word.

        The LLM remains responsible for the rest of the request. A negated or
        competing format, quoted title, or model-declared format issue leaves
        the turn for ordinary validation instead of guessing a value.
        """
        field = self.domain_field
        if (
            not field
            or not self.explicit_domain_surfaces
            or request.reset_constraints
            or any(update.field == field for update in request.updates)
            or any(issue.field == field for issue in request.issues)
        ):
            return None
        visible = message
        for match in re.finditer(r'«[^»]*»|"[^"]*"', visible):
            visible = visible[: match.start()] + " " * (match.end() - match.start()) + visible[match.end() :]
        for update in request.updates:
            if update.field != "seed_title":
                continue
            start = message.casefold().find(update.source_text.casefold())
            if start >= 0:
                visible = visible[:start] + " " * len(update.source_text) + visible[start + len(update.source_text) :]
        found: dict[str, str] = {}
        for token in re.finditer(r"[^\W_]+(?:-[^\W_]+)*", visible, re.UNICODE):
            word = canonical_text(token.group())
            for surface in self.explicit_domain_surfaces:
                value = self.aliases.get(field, {}).get(surface)
                if value is not None and word == canonical_text(surface):
                    clause = re.split(r"[,;:.!?]", visible[: token.start()])[-1]
                    preceding = self.language_profile.tokens(canonical_text(clause))
                    if set(preceding) & set(self.language_profile.negation_tokens):
                        continue
                    tail = re.split(r"[,;:.!?]", visible[token.end() :])[0]
                    following = self.language_profile.tokens(canonical_text(tail))
                    if "не" in following:
                        continue
                    found[value] = token.group()
        if len(found) != 1:
            return None
        value, source = next(iter(found.items()))
        return ConstraintUpdate(field=field, operation="set", value=value, source_text=source)

    def domain_question(self, message: str) -> str:
        for surface, question in self.contextual_domain_questions.items():
            if self.language_profile.matches_surface(canonical_text(message), canonical_text(surface)):
                return question
        return self.field_questions.get(self.domain_field, "Что хотите подобрать?")

    def _is_soft_description(self, field: str, source_text: str, message: str) -> bool:
        """Allow hedged descriptions or a declared subjective quality, never strict requirements."""
        text = canonical_text(message)
        source = canonical_text(source_text)
        if not self.soft_descriptor_pattern or not re.fullmatch(self.soft_descriptor_pattern, source):
            return False
        hedged = any(self.language_profile.matches_surface(text, canonical_text(marker)) for marker in self.soft_request_markers)
        named_soft_quality = any(
            self.language_profile.same_lexeme(source, canonical_text(surface)) for surface in self.soft_rank_aliases
        )
        if not hedged and not named_soft_quality:
            return False
        # A short model citation cannot establish the scope of a requirement.
        # A marker anywhere in the turn makes downgrading unsafe.
        if any(self.language_profile.matches_surface(text, canonical_text(marker)) for marker in self.hard_preference_markers):
            return False
        return not any(
            self.language_profile.matches_surface(text, canonical_text(cue))
            for cue in self.clear_cues.get(field, ())
        )

    @staticmethod
    def _source_is_cited(source_text: str, message: str) -> bool:
        """Require an exact source span, separated from surrounding word text.

        A raw substring check lets fabricated evidence pass when a declared
        surface appears only inside an unrelated longer word. This guard stays
        language-agnostic and runs before semantic resolution.
        """
        source = canonical_text(source_text)
        if not source:
            return False
        return bool(re.search(r"(?<!\w)" + re.escape(source) + r"(?!\w)", canonical_text(message)))

    @staticmethod
    def _masked_titles(message: str, title_spans: tuple[str, ...] = ()) -> str:
        visible = message
        for match in re.finditer(r'«[^»]*»|"[^"]*"', message):
            visible = visible[: match.start()] + " " * (match.end() - match.start()) + visible[match.end() :]
        for title in title_spans:
            match = re.search(re.escape(title), visible, re.IGNORECASE)
            if match:
                visible = visible[: match.start()] + " " * (match.end() - match.start()) + visible[match.end() :]
        return visible

    def _value_mentions(self, field: str, value: str, message: str, title_spans: tuple[str, ...] = ()):
        """Return minimal literal mentions of one declared value outside titles."""
        visible = self._masked_titles(message, title_spans)
        words = list(re.finditer(r"[^\W_]+", visible, re.UNICODE))
        max_words = max(
            (len(self.language_profile.tokens(surface)) for surface in (*self.enums[field], *self.aliases.get(field, {}))),
            default=1,
        )
        mentions = []
        for index, word in enumerate(words):
            candidates = []
            for width in range(1, max_words + 1):
                if index + width > len(words):
                    continue
                end = words[index + width - 1].end()
                evidence = visible[word.start() : end]
                if re.search(r"[,;.!?:]", evidence):
                    continue
                candidate_tokens = self.language_profile.tokens(canonical_text(evidence))
                surfaces = (value, *(surface for surface, target in self.aliases.get(field, {}).items() if target == value))
                if any(
                    len(candidate_tokens) == len(surface_tokens)
                    and all(
                        self.language_profile.same_lexeme(actual, expected)
                        for actual, expected in zip(candidate_tokens, surface_tokens, strict=True)
                    )
                    for surface in surfaces
                    if (surface_tokens := self.language_profile.tokens(canonical_text(surface)))
                ):
                    candidates.append((width, end))
            if not candidates:
                continue
            width, end = min(candidates)
            left = max((part.end() for part in re.finditer(r"[,;.!?:]", visible[: word.start()])), default=0)
            next_boundary = re.search(r"[,;.!?:]", visible[end:])
            right = end + next_boundary.start() if next_boundary else len(visible)
            clause_words = [part for part in words if left <= part.start() and part.end() <= right]
            target = next(position for position, part in enumerate(clause_words) if part.start() == word.start())
            mentions.append((message[word.start() : end], clause_words, target, target + width))
        return mentions

    def _group_bridge(self, field: str, words: list[str]) -> bool:
        """Only declared values and conjunctions may extend an operator's scope."""
        connectors = {"и", "тоже", "также"}
        return all(
            word in connectors or self.resolver.resolve(self._field_spec(field), None, word).status == "canonical"
            for word in words
        )

    def _scoped_operations(self, field: str, words: list[re.Match], start: int, end: int) -> set[str]:
        tokens = [canonical_text(word.group()) for word in words]
        markers = (
            *(("exclude", marker) for marker in self.exclusion_markers),
            *(("include", marker) for marker in self.inclusion_markers),
            *(("keep", marker) for marker in self.keep_markers),
        )
        found = set()
        for operation, marker in markers:
            if operation in {"exclude", "include"} and "или" in tokens:
                continue  # A disjunction does not establish which value to change.
            pattern = [canonical_text(token) for token in self.language_profile.tokens(marker)]
            if not pattern:
                continue
            if pattern == ["не"] and tokens.count("не") > 1:
                continue  # Nested negation has no safe polarity without a full parse.
            for position in range(len(tokens) - len(pattern) + 1):
                if not all(self.language_profile.same_lexeme(actual, expected) for actual, expected in zip(tokens[position:], pattern, strict=False)):
                    continue
                marker_end = position + len(pattern)
                if position < start and marker_end <= start:
                    bridge = tokens[marker_end:start]
                    if self._operator_is_negated(tokens, position) and pattern[0] != "не":
                        continue  # "не исключайте" reverses the operator.
                    if pattern == ["не"]:
                        if bridge not in ([], ["нужен"], ["нужна"], ["нужно"], ["нужны"], ["хочу"]):
                            continue
                    elif not self._group_bridge(field, bridge):
                        continue
                    if bridge and pattern != ["не"] and not self._group_bridge(field, tokens[end:]):
                        continue  # A new predicate after this value ends the shared operator.
                    if "не" in tokens[end:]:
                        continue  # A later negation can reverse the whole request: "без X не хочу".
                    found.add(operation)
                elif position >= end:
                    if pattern in (["не"], ["без"]):
                        continue
                    bridge = tokens[end:position]
                    if bridge not in ([], ["тоже"], ["также"], ["снова"], ["опять"]) and not self._group_bridge(field, bridge):
                        continue
                    if self._operator_is_negated(tokens, position) and pattern[0] != "не":
                        continue
                    found.add(operation)
        return found

    @staticmethod
    def _operator_is_negated(tokens: list[str], position: int) -> bool:
        # An unknown bridge after negation cannot justify a positive operation.
        # The caller may ask for clarification instead of silently changing state.
        return "не" in tokens[:position]

    def _operation_evidence(
        self, update: ConstraintUpdate, message: str, operation: str, title_spans: tuple[str, ...] = ()
    ) -> str | None:
        """Recover a value citation only when a local operator governs it."""
        if update.field not in self.enums or not isinstance(update.value, str):
            return None
        matches = [
            evidence
            for evidence, words, start, end in self._value_mentions(update.field, update.value, message, title_spans)
            if self._scoped_operations(update.field, words, start, end) == {operation}
        ]
        return matches[0] if matches else None

    def _negated_exclusion_evidence(self, update: ConstraintUpdate, message: str, titles: tuple[str, ...]) -> str | None:
        """Recognize a declared 'do not exclude X' immediately governing X."""
        if update.field not in self.enums or not isinstance(update.value, str):
            return None
        for evidence, words, start, end in self._value_mentions(update.field, update.value, message, titles):
            if self._scoped_operations(update.field, words, start, end) != {"include"}:
                continue
            tokens = [canonical_text(word.group()) for word in words]
            for marker in self.negated_exclusion_markers:
                pattern = [canonical_text(token) for token in self.language_profile.tokens(marker)]
                for position in range(len(tokens) - len(pattern) + 1):
                    if not all(
                        self.language_profile.same_lexeme(actual, expected)
                        for actual, expected in zip(tokens[position : position + len(pattern)], pattern, strict=True)
                    ):
                        continue
                    marker_end = position + len(pattern)
                    if marker_end <= start and self._group_bridge(update.field, tokens[marker_end:start]):
                        return evidence
                    if position >= end and self._group_bridge(update.field, tokens[end:position]):
                        return evidence
        return None

    def _operator_disjunction(self, update: ConstraintUpdate, message: str, titles: tuple[str, ...]) -> bool:
        """A value inside 'exclude/include X or Y' cannot become a required X."""
        if update.field not in self.enums or not isinstance(update.value, str):
            return False
        markers = (*self.exclusion_markers, *self.inclusion_markers)
        for _, words, _, _ in self._value_mentions(update.field, update.value, message, titles):
            tokens = [canonical_text(word.group()) for word in words]
            if "или" not in tokens:
                continue
            for marker in markers:
                pattern = [canonical_text(token) for token in self.language_profile.tokens(marker)]
                if pattern and any(
                    all(self.language_profile.same_lexeme(actual, expected)
                        for actual, expected in zip(tokens[position : position + len(pattern)], pattern, strict=True))
                    for position in range(len(tokens) - len(pattern) + 1)
                ):
                    return True
        return False

    def reconcile_operation_evidence(
        self,
        request: StructuredRequest,
        message: str,
        *,
        excluded: set[tuple[str, object]] | None = None,
        retained: set[tuple[str, object]] | None = None,
    ) -> StructuredRequest:
        """Correct model citations and a mistaken positive set for a revoked ban."""
        updates = []
        titles = tuple(update.source_text for update in request.updates if update.field == "seed_title")
        for update in request.updates:
            operation = update.operation
            if operation in {"exclude", "include"} and self._operation_evidence(update, message, "keep", titles) is not None:
                operation = "set"
            if operation == "set" and self._negated_exclusion_evidence(update, message, titles) is not None:
                operation = "include"
            if (
                operation == "set"
                and self._operation_evidence(update, message, "exclude", titles) is not None
                and any(
                    marker != "не" and self.language_profile.matches_surface(canonical_text(message), canonical_text(marker))
                    for marker in self.exclusion_markers
                )
                and not self.detect_polarity_conflicts(StructuredRequest(updates=[update]), message).issues
            ):
                operation = "exclude"
            if (
                operation == "set"
                and (update.field, update.value) in (excluded or set())
                and self._operation_evidence(update, message, "include", titles) is not None
            ):
                operation = "include"
            evidence = self._operation_evidence(update, message, operation, titles) if operation in {"exclude", "include"} else None
            updates.append(update.model_copy(update={"operation": operation, "source_text": evidence or update.source_text}))
        return request.model_copy(update={"updates": updates})

    def detect_polarity_conflicts(self, request: StructuredRequest, message: str) -> StructuredRequest:
        """Block the same declared value requested and excluded in one turn."""
        if not self.exclusion_markers:
            return request
        issues = list(request.issues)
        titles = tuple(update.source_text for update in request.updates if update.field == "seed_title")
        for update in request.updates:
            if update.operation not in {"set", "exclude", "include"} or update.field not in self.enums or not isinstance(update.value, str):
                continue
            positive: str | None = None
            negative = False
            for evidence, words, start, end in self._value_mentions(update.field, update.value, message, titles):
                scope = self._scoped_operations(update.field, words, start, end)
                if scope == {"exclude"}:
                    negative = True
                elif not scope or scope in ({"include"}, {"keep"}):
                    positive = evidence
            if positive and negative:
                issues.append(
                    InterpretationIssue(
                        kind="conflict",
                        field=update.field,
                        value=str(update.value),
                        source_text=positive,
                        message="Вы одновременно попросили это условие и исключили его. Уточните, какое пожелание сохранить.",
                    )
                )
        return request.model_copy(update={"issues": issues, "clarification_required": bool(issues) or request.clarification_required})

    def _value_is_negated(self, field: str, value: object, source_text: str) -> bool:
        """Check polarity immediately around the cited value, not the whole span.

        A structured extractor may cite the complete user phrase for one field.
        A sentence-level ``has_negation`` check would then reject an unrelated
        positive value such as ``film`` in ``film, not dark``.
        """
        if isinstance(value, str):
            surfaces = [value, *self.aliases.get(field, {}).keys()]
        else:
            surfaces = [alias for alias, target in self.scalar_aliases.get(field, {}).items() if target == value]
        surface_tokens = [tuple(self.language_profile.tokens(canonical_text(surface))) for surface in surfaces]
        surface_tokens = [tokens for tokens in surface_tokens if tokens]
        if not surface_tokens:
            return False
        text_tokens = self.language_profile.tokens(canonical_text(source_text))
        negations = {canonical_text(token) for token in self.language_profile.negation_tokens}
        for index, token in enumerate(text_tokens):
            if canonical_text(token) not in negations:
                continue
            for candidate in surface_tokens:
                end = index + 1 + len(candidate)
                if end > len(text_tokens):
                    continue
                if all(
                    self.language_profile.same_lexeme(canonical_text(actual), canonical_text(expected))
                    for actual, expected in zip(text_tokens[index + 1 : end], candidate, strict=False)
                ):
                    return True
        return False

    def normalize(self, request: StructuredRequest, message: str) -> StructuredRequest:
        """Return only evidence-verified updates in the domain's canonical form.

        Canonical state and the legacy Query projection share this normalization.
        Rejected updates become blocking issues; session state is not mutated.
        """
        canonical_updates = []
        issues = list(request.issues)
        titles = tuple(update.source_text for update in request.updates if update.field == "seed_title")
        for update in request.updates:
            name, value = update.field, update.value
            nli_enum = self.context_verifier is not None and name in self.enums and update.operation in {"set", "exclude", "include"}
            if not nli_enum and update.operation == "set" and self._operator_disjunction(update, message, titles):
                issues.append(self._issue(name, update.source_text, kind="ambiguity"))
                continue
            if name not in self.model.model_fields or name == "intent":
                issues.append(self._issue(name, value))
                continue
            if not self._source_is_cited(update.source_text, message):
                issues.append(self._issue(name, value, kind="ambiguity"))
                continue
            if not nli_enum and update.operation == "set" and self._explicit_no_preference(name, update.source_text):
                canonical_updates.append(update.model_copy(update={"operation": "clear", "value": None}))
                continue
            if update.operation == "clear" and not self._explicit_no_preference(name, update.source_text):
                source = canonical_text(update.source_text)
                if self._clear_citation_conflicts(source):
                    issues.append(self._issue(name, update.source_text, kind="ambiguity"))
                    continue
                if name in self.clear_cues and self._clear_fields(source) != {name}:
                    issues.append(self._issue(name, update.source_text, kind="ambiguity"))
                    continue
                if (
                    self.clear_markers
                    and name in self.clear_cues
                    and not any(self.language_profile.matches_surface(source, canonical_text(marker)) for marker in self.clear_markers)
                ):
                    issues.append(self._issue(name, update.source_text, kind="ambiguity"))
                    continue
                if name in self.enums:
                    surfaces = (*self.enums[name], *self.aliases.get(name, {}))
                    if any(self.language_profile.matches_surface(source, surface) for surface in surfaces):
                        issues.append(self._issue(name, update.source_text, kind="ambiguity"))
                        continue
            # An exclusion must cite its negation; a positive value alone
            # cannot justify excluding it.
            if not nli_enum and (
                (update.operation == "exclude" and not self.exclusion_markers and not self.language_profile.has_negation(update.source_text))
                or (
                    update.operation in {"exclude", "include"}
                    and (self.exclusion_markers or self.inclusion_markers)
                    and not self._operation_evidence(update, message, update.operation)
                )
            ):
                issues.append(self._issue(name, update.source_text, kind="ambiguity"))
                continue
            if type(value) is int and not re.search(
                r"(?<![\d.,])" + re.escape(canonical_text(update.source_text)) + r"(?!\d|[.,]\d)",
                canonical_text(message),
            ):
                issues.append(self._issue(name, update.source_text, kind="ambiguity"))
                continue
            if update.operation != "clear":
                # Generic schema-driven resolution: the LLM proposed a canonical
                # value and cited the evidence it read it from. The adapter checks
                # coherence with declared surfaces instead of re-parsing language.
                if not nli_enum and (
                    update.operation == "set"
                    and (name in self.enums or value is True)
                    and self._value_is_negated(name, value, update.source_text)
                ):
                    issues.append(self._issue(name, update.source_text, kind="ambiguity"))
                    continue
                if nli_enum:
                    allowed = ("set", "exclude", "include") if name in self.exclusions or name in self.canonical_exclusions else ("set",)
                    resolution = self.context_verifier.verify(EvidenceInput(
                        message=message,
                        update=update,
                        spec=self._field_spec(name),
                        field_label=self.field_labels.get(name, name),
                        allowed_operations=allowed,
                        is_preference=name in self.preference_fields,
                    )).resolution
                elif name in self.enums:
                    resolution = self.resolver.resolve(self._field_spec(name), value, update.source_text)
                else:
                    resolution = resolve_scalar(self._field_spec(name), value, update.source_text, profile=self.language_profile)
                if resolution.status != "canonical":
                    kind = "ambiguity" if resolution.status == "ambiguous" else "unsupported_constraint"
                    issues.append(self._issue(name, resolution.evidence or value, kind=kind, source_text=update.source_text))
                    continue
                value = resolution.value
            if update.operation in ("exclude", "include") and name not in self.exclusions and name not in self.canonical_exclusions:
                issues.append(self._issue(name, value))
                continue
            canonical_updates.append(update.model_copy(update={"value": None if update.operation == "clear" else value}))
        if request.clarification_required and not issues and not request.updates:
            issues.append(self._issue("request", message, kind="ambiguity"))
        unique = {(issue.kind, issue.field, issue.value): issue for issue in issues}
        return request.model_copy(update={"updates": canonical_updates, "issues": list(unique.values())})

    def apply(
        self, request: StructuredRequest, previous: BaseModel, message: str, unresolved: list[InterpretationIssue] | None = None
    ) -> tuple[BaseModel, list[InterpretationIssue]]:
        request = self.normalize(request, message)
        values = previous.model_dump()
        updates = {}
        issues = list(request.issues)
        changed = set()
        for update in request.updates:
            name, value = update.field, update.value
            if update.operation in ("exclude", "include"):
                target = self.exclusions.get(name)
                if target is None:
                    # The legacy request DTO need not express every canonical
                    # constraint. A declared canonical-only exclusion is
                    # carried by ConstraintState and residual filtering.
                    if name not in self.canonical_exclusions:
                        issues.append(self._issue(name, value))
                        continue
                    changed.add(name)
                    continue
                existing = list(updates.get(target, values[target]))
                if update.operation == "exclude" and value not in existing:
                    existing.append(value)
                elif update.operation == "include" and value in existing:
                    existing.remove(value)
                updates[target] = existing
            else:
                if name in updates and updates[name] != value and update.operation == "set":
                    issues.append(self._issue(name, value, kind="conflict"))
                    continue
                updates[name] = None if update.operation == "clear" else value
            changed.add(name)
        # Schema-defined domain changes reset old fields without interpreting language.
        domain_changed = (
            self.domain_field
            and values.get(self.domain_field) is not None
            and updates.get(self.domain_field) is not None
            and updates[self.domain_field] != values[self.domain_field]
        )
        if domain_changed:
            values = self.model().model_dump()
        values.update(updates)
        for source_field in changed:
            source_value = values.get(source_field)
            for target_field, target_value in self.implied_values.get(source_field, {}).get(source_value, {}).items():
                if target_field not in changed and values.get(target_field) is None:
                    values[target_field] = target_value
        if request.intent is not None:
            values["intent"] = request.intent
        for field, excluded in self.exclusions.items():
            if values.get(field) is not None and values[field] in values.get(excluded, []):
                issues.append(self._issue(field, values[field], kind="conflict"))
                values[field] = previous.model_dump()[field]
                values[excluded] = previous.model_dump()[excluded]
        for issue in unresolved or []:
            if issue.field not in changed and not domain_changed:
                issues.append(issue)
        # A bare clarification flag blocks an empty patch, not verified updates.
        # Explicit and locally detected issues remain blocking in either case.
        if request.clarification_required and not issues and not request.updates:
            issues.append(self._issue("request", message, kind="ambiguity"))
        try:
            query = self.model.model_validate(values, strict=True)
        except ValidationError as exc:
            query = previous
            issues.extend(self._issue(str(error["loc"][0]), "значение вне schema") for error in exc.errors())
        unique = {(issue.kind, issue.field, issue.value): issue for issue in issues}
        return query, list(unique.values())
