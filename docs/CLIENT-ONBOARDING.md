# Подключение нового клиента

Для проверки заменяемости рекомендательной платформы я добавил второй
демонстрационный домен — электронику. Он использует собственные схему запроса,
каталог и provider, но проходит через общий диалоговый граф. Контракт настоящего
Recsflow для проекта не предоставлен.

## Что нужно реализовать

1. Опишите схему запроса в Pydantic и создайте `SchemaRequestAdapter` с названиями
   полей, единицами измерения и правилами исключений для своего домена.
2. Создайте `DomainSpec`: перечислите `FieldSpec`, `item_field` и
   `default_operator`. Например, `max_price` проецируется на `price_rub` с `lte`.
3. Реализуйте provider с `retrieve(user_id, query, limit)` и `lookup(ids)`.
   Provider может сузить выдачу, но обязательный фильтр всё равно применяется
   после lookup.
4. Передайте функцию преобразования карточки в `build_domain_workflow`. Она
   переводит поля источника в атрибуты для фильтрации и объяснений. Например,
   признак `portable` для электроники вычисляется как `weight_kg <= 1.5`.
5. Добавьте интеграционный тест с уточнением, выдачей и отсутствием результатов. Тест
   `tests/integration/test_second_customer_portability.py` — исполняемый пример:
   категория ноутбука, цена и переносимость проверяются общими компонентами.

## Manifest и создание сервиса

Проверка подключения связывает YAML manifest с этим сервисом через
`load_customer_manifest()` и `onboard_customer()`. Они сверяют домен и схему,
полноту сопоставления полей, заявленные возможности и наличие методов provider.
`portable` в manifest указывает на вычисленный атрибут `portable`; преобразование
из `weight_kg` остаётся в функции преобразования карточки клиента.

После определения схемы, provider и преобразования карточек сервис создаётся так:

```python
from pathlib import Path
from recagent.onboarding import load_customer_manifest, onboard_customer

service = onboard_customer(
    manifest=load_customer_manifest(Path("configs/customers/electronics.yaml")),
    domain=domain_spec,
    provider=customer_provider,
    backend=structured_backend,
    adapter=request_adapter,
    supported_capabilities={"category_filter", "max_price_lte", "portable_filter"},
    item_projection=customer_item_projection,
)
result = service.chat(user_id="customer-user", message="...")
```

`service.execute("history")` возвращает `unsupported_operation`, если операция
не объявлена. Ошибки связи/таймауты provider или backend возвращаются как
`upstream_unavailable`, без выдачи неподтверждённых объектов. Недостающий lookup
не позволяет исполнить recommend. Это поведение компонента; соответствие
HTTP-статусам определяется адаптером конкретного сервиса.

## Проверка

Проверенная цепочка: YAML → проверка полей и возможностей → «недорогой ноутбук» → вопрос
о бюджете → «100000 рублей» → только ноутбук в пределах цены → lookup.
Подмешанный дешёвый планшет отфильтрован. Проверены неверное сопоставление полей,
отсутствующие методы, неподдерживаемые операции, таймауты и ошибки HTTP-соединения.
Проверка воспроизводится командой:

```powershell
python -m pytest -q tests/integration/test_second_customer_portability.py
```

В тестах используется управляемый backend структурированного разбора.
Они проверяют связность компонентов и сохранение условий между ходами.
Качество реальной LLM на запросах об электронике требует отдельного измерения.

`GenericWorkflowResponse` возвращает запрос, идентификаторы объектов и trace.
Сериализация API основного демо сохраняет совместимость с `ChatResponse`.
Для внешнего клиента отдельно потребуются авторизация, постоянное хранилище
и проверка фактического API по [интеграционной инструкции](contract/INTEGRATION-CHECKLIST.md).
