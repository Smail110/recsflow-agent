# Воспроизведение RecAgent

Здесь собраны команды для установки из чистой копии, запуска демо и повторения
оценки. Основной маршрут — `workflow-v2`, модель — `qwen3:8b`. Демонстрационный
каталог и диалоги оценки синтетические. Происхождение данных описано в
[DATA-PROVENANCE.md](DATA-PROVENANCE.md), результаты — в [EVALUATION.md](EVALUATION.md).

## Установка Python

Поддерживается Python 3.11 или 3.12. Не переносите готовую `.venv` из другой
папки: на Windows она хранит путь к установленному интерпретатору. В PowerShell
из корня отдельной копии:

```powershell
python --version
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.venv\Scripts\python.exe -m pip install -e . --no-deps
.venv\Scripts\python.exe -m pip check
```

`pip check` должен вывести `No broken requirements found`. Для установки
нужен доступ к Python-пакетам. Исследовательские зависимости Hugging Face,
CUDA и автоматической разметки обычному демо не требуются.

## Проверки без LLM

```powershell
.venv\Scripts\python.exe -m pytest -q tests/unit/test_product_contract_data.py tests/unit/test_observable_product.py tests/integration/test_api_http.py
.venv\Scripts\python.exe -m scripts.validate_contract
.venv\Scripts\python.exe -m scripts.evaluate_llm_first_product --validate-only --label after --implementation workflow-v2 --output report/dev-validation-local.json
.venv\Scripts\python.exe -m scripts.run_demo_notebook --mode rules --output report/demo-rules-local.ipynb
```

`rules` проверяет отдельный ограниченный маршрут, а не качество `qwen3:8b`.
Для полной проверки поставляемых компонентов:

```powershell
.venv\Scripts\python.exe -m pytest -q --ignore-glob='*test_semantic_lora*' --ignore-glob='*test_semantic_extraction_blind*'
.venv\Scripts\python.exe -m ruff check src scripts evals tests app.py
```

Исключённые исследовательские тесты требуют отдельных данных обучения и
не входят в приёмку демо. Сохраняйте результат вместе с версией Python,
Git commit и командой запуска: число тестов может меняться.

## Локальная модель и notebook

Установите Ollama, запустите её на хосте и загрузите модель. Новый inference
может отличаться от сохранённого; идентификатор модели и фактический digest
нужно фиксировать вместе с результатом.

```powershell
ollama pull qwen3:8b
ollama list
$env:RECAGENT_MODE='ollama'
$env:OLLAMA_MODEL='qwen3:8b'
$env:OLLAMA_URL='http://127.0.0.1:11434'
$env:RECAGENT_RESPONSE_PLANNER='llm'
.venv\Scripts\python.exe -m scripts.run_demo_notebook --mode ollama --output report/demo-ollama-local.ipynb
.venv\Scripts\python.exe -m scripts.smoke_public_tree
```

Notebook проверяет непустую подходящую выдачу и `mode=ollama`;
`smoke_public_tree` поднимает отдельные процессы mock-платформы и API на
свободных локальных портах и проверяет HTTP `/health`, `/ready`, `/v1/chat`,
пять карточек и отсутствие fallback. Успешный rules fallback не заменяет
эту проверку.

## Docker и HTTP

Compose поднимает mock-платформу, API и Streamlit. Ollama работает на хосте;
контейнеры обращаются к `host.docker.internal:11434`.

```powershell
docker compose config
docker compose build
docker compose up -d
docker compose ps
Invoke-RestMethod http://127.0.0.1:8090/health
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/ready
Invoke-WebRequest http://127.0.0.1:8501/_stcore/health
$body = @{user_id='demo'; message='Хочу курс Python с практикой.'} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/chat -ContentType 'application/json; charset=utf-8' -Body ([System.Text.Encoding]::UTF8.GetBytes($body))
```

В ответе проверьте `state=recommend`, подходящие карточки курсов,
`mode=ollama`, отсутствие fallback и telemetry. HTTP 200 без подходящей
выдачи не считается успешной рекомендацией. При ошибке сборки Docker из
кириллического каталога в PowerShell:

```powershell
$env:COMPOSE_BAKE='false'
$env:DOCKER_BUILDKIT='0'
docker compose up --build -d
```

Если в системе уже работает Compose-проект, запускайте чистую копию с другим
project name и host ports: одинаковые имя и порты не дают независимой проверки
и могут заменить старые контейнеры. Операционная диагностика — в
[runbook.md](runbook.md).

## Данные и повторение оценки

DEV: `data/product_llm_first_dev.json`, 20 диалогов, каталог seed 42.
CONTRACT: `data/product_contract_ru_v1.json` и соответствующий manifest,
12 диалогов, каталог seed 137. Входы и критерии должны совпасть по хешам;
отсутствующий файл нельзя заменить новой похожей синтетикой и назвать тем же
результатом. CONTRACT можно заново собрать по
[описанию генератора](data/PRODUCT-CONTRACT.md). Для нового DEV inference:

```powershell
.venv\Scripts\python.exe -m scripts.evaluate_llm_first_product --label after --implementation workflow-v2 --workflow-config configs/workflow-v2.yaml --output report/dev-local.json
.venv\Scripts\python.exe -m scripts.evaluate_product_contract --cohort data/product_contract_ru_v1.json --manifest data/product_contract_ru_v1.manifest.json --mode ollama --implementation workflow-v2 --output report/contract-local.json
```

Это новые прогоны, а не побайтовое восстановление сохранённых ответов LLM.
Сравнивайте с [описанием метрик](EVALUATION.md), сохраняя model digest,
параметры, окружение, сырые ответы и хеши входов.

Для повторения нагрузочного опыта нужен запущенный API в режиме Ollama.
Команды выполняются последовательно; параллельный запуск других задач на
той же модели изменит задержки:

```powershell
.venv\Scripts\python.exe -m scripts.load_test --requests 50 --concurrency 1 --output report/load-c1-local.json
.venv\Scripts\python.exe -m scripts.load_test --requests 50 --concurrency 2 --output report/load-c2-local.json
.venv\Scripts\python.exe -m scripts.load_test --requests 50 --concurrency 4 --output report/load-c4-local.json
```

В JSON сохраняются p50/p95, число HTTP-ответов, непустых выдач, вызовов LLM и
токенов. Сценарий повторяет один запрос и не оценивает разнообразие диалогов.
Фактическая денежная стоимость остаётся неизвестной. Для условного расчёта
выделенной машины задайте ставку в рублях за час:

```powershell
.venv\Scripts\python.exe -m scripts.estimate_compute_cost `
  --load report/load-c1-local.json --load report/load-c2-local.json --load report/load-c4-local.json `
  --rates 50 100 200 --output report/compute-cost-local.json
```

Ставки приведены для анализа чувствительности, а не как тариф поставщика.
Расчёт использует длительность всего окна нагрузки: ставка × секунды / 3600.
Перекрывающиеся запросы не суммируются как часы аренды. Простой, загрузка модели
до измерения, хранение и сопровождение не включены. Старый флаг load runner
`--gpu-hour-cost` считает другую величину — сумму учтённого времени инференса;
её нельзя выдавать за счёт за выделенную машину.

## Оценка с LLM-симулятором и судьёй

Этот сценарий запускает восемь диалогов из открытого DEV-набора. Отдельные
роли симулируют пользователя, проверяют смысл ответа и оценивают результат;
для всех ролей используется `qwen3:8b`. Нужна работающая Ollama.

```powershell
$modelDigest = '500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41'
.venv\Scripts\python.exe -m scripts.evaluate_llm_protocol `
  --manifest artifacts/eval-v2-20260914/recagent-eval-v2.0-dev.manifest.json `
  --implementation workflow-v2 --workflow-config configs/workflow-v2.yaml `
  --agent-mode ollama --agent-model qwen3:8b --agent-digest $modelDigest `
  --simulator-model qwen3:8b --simulator-digest $modelDigest `
  --semantic-validator-model qwen3:8b --semantic-validator-digest $modelDigest `
  --judge-model qwen3:8b --judge-digest $modelDigest `
  --limit 8 --question-policy adaptive --max-clarifications 3 `
  --seed 42 --temperature 0 --num-predict 700 `
  --cache-mode record --cache-dir runtime/llm-protocol-local `
  --output report/llm-protocol-local.json
```

Digest закрепляет использованную модель. При несовпадении скрипт остановится:
если вы сознательно выбираете другую сборку, укажите её фактический digest
и считайте результат отдельным опытом. Ответы ролей и их идентификаторы
сохраняются в указанном кэше; состав диалогов фиксируется до обращения к LLM.

Это новый прогон текущего кода: сохранённый строгий результат **2/8** от 27.09 не
гарантирован. Исходный кэш ответов в поставку не включён. После успешной
записи свой прогон можно повторить без новых вызовов LLM, заменив
`--cache-mode record` на `--cache-mode require_cache` и сохранив остальные
параметры и файлы кэша. Неполный кэш не считается успешным повторением.
Код выхода 1 также возможен при завершённом протоколе, если есть неуспешные
эпизоды. Смотрите отдельно `status`, `protocol_complete_count` и `success_count`.

## Граница внешней интеграции

`docker compose` использует локальный HTTP mock. Настоящий Recsflow требует
endpoint, credentials, подтверждённой схемы, тестового каталога и staging.
Предложенный OpenAPI не доказывает совместимость с внешней платформой. Сессии
сейчас находятся в памяти одного процесса; перезапуск очищает их.
