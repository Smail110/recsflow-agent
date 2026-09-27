"""The existing media/course demo schema; no language rules in universal core."""

from ..models import Query
from ..request_mapping import SchemaRequestAdapter
from ..resolution import LanguageProfile
from .base import DomainSpec, FieldSpec

NO_PREFERENCE_MARKERS = ("любой", "любого", "любая", "любые", "не важно", "неважно", "сколько угодно", "без ограничений")


def request_adapter() -> SchemaRequestAdapter:
    return SchemaRequestAdapter(
        Query,
        domain_field="kind",
        exclusions={"genre": "excluded_genres"},
        language_profile=LanguageProfile(
            negation_tokens=("не", "без"),
            numeric_words=(
                ("один", 1),
                ("одна", 1),
                ("одно", 1),
                ("одного", 1),
                ("одной", 1),
                ("одну", 1),
                ("два", 2),
                ("две", 2),
                ("двух", 2),
                ("три", 3),
                ("трёх", 3),
                ("трех", 3),
                ("четыре", 4),
                ("четырёх", 4),
                ("четырех", 4),
                ("пять", 5),
                ("пяти", 5),
            ),
        ),
        field_labels={"kind": "формат", "genre": "жанр или тема", "level": "уровень", "tone": "тон"},
        display_labels={
            "kind": "формат",
            "genre": "жанр или тема",
            "level": "уровень",
            "tone": "настроение",
            "max_seasons": "число сезонов",
            "max_minutes": "длительность",
            "seasons": "число сезонов",
            "minutes": "длительность",
            "episodes": "число серий",
            "practical": "практические задания",
            "seed_title": "название",
            "request": "пожелание",
        },
        field_questions={
            "kind": "Что подобрать: фильм, сериал или курс?",
            "genre": "Какой жанр или тему вы предпочитаете?",
            "tone": "Какое настроение вам ближе: лёгкое, мрачное или нейтральное?",
            "max_seasons": "Сколько сезонов максимум вам подойдёт?",
            "max_minutes": "Какая максимальная длительность вам подойдёт?",
            "level": "Для какого уровня подготовки нужен курс: начального, среднего или продвинутого?",
            "practical": "Нужны ли в курсе практические задания?",
            "seed_title": "На какое название ориентироваться?",
        },
        contextual_domain_questions={
            "посмотреть": "Что вам подобрать: фильм или сериал?",
            "смотреть": "Что вам подобрать: фильм или сериал?",
        },
        no_preference_markers=dict.fromkeys(("genre", "tone", "max_seasons", "max_minutes", "level", "practical"), NO_PREFERENCE_MARKERS),
        clear_cues={
            "genre": ("жанр", "тема"),
            "tone": ("тон", "тону", "тона", "настроение"),
            "max_seasons": ("сезон",),
            "max_minutes": ("длительность", "минут", "час", "часам", "часов"),
            "level": ("уровень",),
            "practical": ("практика", "задания"),
        },
        clear_markers=("снять", "снимите", "убрать", "уберите", "отменить", "отмените", "не ограничивай", "не ограничивайте"),
        keep_markers=("оставьте", "оставить", "сохраните", "сохранить"),
        soft_description_fields={"genre", "tone"},
        soft_rank_fields={"genre"},
        soft_rank_aliases={
            "смешной": ("genre", "комедия"),
            "весёлый": ("genre", "комедия"),
            "забавный": ("genre", "комедия"),
        },
        soft_request_markers=(
            "какой-нибудь", "какой нибудь", "какой-то", "что-нибудь", "что-то",
            "такой", "типа", "вроде", "примерно", "похожий",
        ),
        soft_descriptor_pattern=r"[а-яё-]+(?:ый|ий|ой|ая|ое|ее|ые|ие|ого|ему|ую|ыми|их|ых)(?:\s+[а-яё-]+(?:ый|ий|ой|ая|ое|ее|ые|ие|ого|ему|ую|ыми|их|ых)){0,2}",
        explicit_domain_surfaces=("фильм", "фильмы", "кино", "сериал", "сериалы", "курс", "курсы"),
        hard_preference_markers=("только", "строго", "обязательно", "исключительно", "без", "не", "должен", "должна", "должно", "должны", "именно"),
        exclusion_markers=("не", "без", "исключить", "исключите", "исключаю", "запрет на", "запретить", "запретите", "не предлагайте"),
        inclusion_markers=("разрешаю", "разрешить", "разрешите", "вернуть", "верните", "снова можно", "снова допускается", "не исключайте", "не исключать"),
        negated_exclusion_markers=("не исключайте", "не исключать"),
        aliases={
            "kind": {
                "сериал": "series",
                "сериалы": "series",
                "фильм": "film",
                "фильмы": "film",
                "кино": "film",
                "курс": "course",
                "курсы": "course",
                "python": "course",
                "питон": "course",
                "машинное обучение": "course",
                "ml": "course",
            },
            "genre": {
                "питон": "python",
                "питону": "python",
                "машинному обучению": "машинное обучение",
                "ml": "машинное обучение",
                "детективный": "детектив",
                "детективы": "детектив",
                "детектива": "детектив",
                "детективов": "детектив",
                "комедию": "комедия",
                "драму": "драма",
                "драмы": "драма",
                "фантастический": "фантастика",
            },
            "level": {
                "новичка": "начальный",
                "новичок": "начальный",
                "с нуля": "начальный",
                "начинающий": "начальный",
                "beginner": "начальный",
                "advanced": "продвинутый",
                "опытный": "продвинутый",
                "продвинутого": "продвинутый",
            },
            "tone": {"лёгкое": "лёгкий", "легкое": "лёгкий", "лёгкую": "лёгкий", "мрачное": "мрачный", "мрачного": "мрачный"},
        },
        scalar_aliases={
            "practical": {
                "практика": True,
                "практикой": True,
                "практическое": True,
                "практическими заданиями": True,
                "заданиями": True,
                "с заданиями": True,
                "без воды": True,
                "без практики": False,
                "только теория": False,
                "только теорию": False,
            },
            "max_minutes": {"полчаса": 30, "час": 60},
            "max_seasons": {"один сезон": 1},
        },
        numeric_units={
            "max_minutes": {"минут": 1, "минуты": 1, "минуту": 1, "мин": 1, "час": 60, "часа": 60, "часов": 60},
            "max_seasons": {"сезон": 1, "сезона": 1, "сезонов": 1},
        },
        implied_values={
            "genre": {
                "python": {"kind": "course"},
                "машинное обучение": {"kind": "course"},
            }
        },
        constraint_operators={"max_seasons": "lte", "max_minutes": "lte"},
        constraint_item_fields={"max_seasons": "seasons", "max_minutes": "minutes"},
        canonical_exclusions={"tone"},
    )


def domain_spec() -> DomainSpec:
    """The demo contract shared by the experimental interpretation transport.

    Values and surface forms remain owned by ``request_adapter``.  This spec
    declares only field shape, canonical operations and item projection, so it
    cannot become a second language parser.
    """

    return DomainSpec(
        id="demo-media-course",
        version="1",
        request_schema="recagent.models.Query",
        fields=(
            FieldSpec(name="kind", value_type="enum"),
            FieldSpec(name="genre", value_type="enum", operators=("eq", "neq")),
            FieldSpec(name="tone", value_type="enum", operators=("eq", "neq")),
            FieldSpec(
                name="max_seasons",
                value_type="integer",
                operators=("eq", "lte"),
                default_operator="lte",
                item_field="seasons",
                unit="season",
            ),
            FieldSpec(
                name="max_minutes",
                value_type="integer",
                operators=("eq", "lte"),
                default_operator="lte",
                item_field="minutes",
                unit="minute",
            ),
            FieldSpec(name="level", value_type="enum"),
            FieldSpec(name="practical", value_type="boolean"),
            FieldSpec(name="seed_title", value_type="string"),
        ),
        field_labels=(
            ("kind", "формат"),
            ("genre", "жанр или тема"),
            ("tone", "тон"),
            ("level", "уровень"),
        ),
        item_projection=(("max_seasons", "seasons"), ("max_minutes", "minutes")),
    )
