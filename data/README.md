# Данные для демонстрации и проверок

В этой папке находятся синтетические тестовые примеры. Реального каталога
Recsflow и пользовательских логов здесь нет. Внешние источники перечислены
ссылками в [описании данных](../docs/DATA-PROVENANCE.md); сырые корпуса
скачиваются отдельно и не входят в репозиторий.

## Основные входы оценки

| Файл | Назначение |
|---|---|
| `product_llm_first_dev.json` | 20 фиксированных сценариев полного диалога. Основной DEV-контроль. |
| `product_contract_ru_v1.json` и manifest | 12 сценариев, 20 ходов: исправления, исключения и продолжение поиска. |
| `model_extraction_*.json`, `e7-*.json` | Компонентные примеры структурированного разбора и проверки условий. |
| `dialogues.jsonl` | Простой синтетический набор из 160 строк для проверки классификации intent. |

Каталог создаёт `src/recagent/catalog/generator.py`. Seed 42 используется
в демо и DEV, seed 137 — в CONTRACT. Проверяемые ответы задаются сценариями,
а не извлекаются из ответа агента. Команды сборки CONTRACT и проверка хешей —
в [описании набора](../docs/data/PRODUCT-CONTRACT.md).

## Генерация примеров intent

Для побайтового воспроизведения `dialogues.jsonl` предусмотрен отдельный
генератор с seed 42 и 40 строками на каждый из четырёх intent:

```powershell
python -m scripts.reproduce_legacy_dialogues `
  --reference data/dialogues.jsonl `
  --output artifacts/reproducibility/legacy-dialogues.jsonl
```

Скрипт проверяет SHA-256 содержимого с переводами строк LF:
`b03da67e46a8a3009332ea616db9bb0cd8d405f14d25441adbbf070080397877`.
CRLF нормализуется только при этой проверке; результат записывается с LF.
В наборе есть повторы и нет разделения по семействам формулировок, поэтому
он не подходит для независимой оценки обучения. Основной продуктовый
success rate на нём не считается.

## Примеры с разделением по шаблонам

Новые синтетические smoke-данные создаёт `scripts/prepare_hf_dataset.py`:

```powershell
python -m scripts.prepare_hf_dataset `
  --output data/dialogues-grouped.jsonl `
  --seed 42 --examples-per-label 40 --validation-fraction 0.25
```

Эта схема дополнительно содержит `template_family_id`, `template_version`,
`normalized_text_sha256` и `split`. Train/validation разделяются по целым template
families и проходят проверки отсутствия exact/normalized duplicate text. Это
устраняет прямое пересечение template families, но синтетические строки всё равно
не являются независимыми пользовательскими диалогами и не подтверждают качество
на реальном трафике.
