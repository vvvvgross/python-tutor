# Виртуальный тьютор Python

Веб-приложение для персонализированного обучения программированию на Python. Пользователь формулирует учебную цель, выбирает уровень подготовки, получает учебный план, изучает теорию, решает задания в редакторе кода, запускает автопроверку и общается с ИИ-тьютором.

Проект использует FastAPI, Docker sandbox для безопасного запуска пользовательского кода, WebSocket-чат, SRL-метрики саморегулируемого обучения и педагогические схемы адаптации подсказок.

## Возможности

- Генерация учебного плана под цель пользователя.
- Теоретический блок с примерами и мини-проверкой понимания.
- Практические задания с автотестами.
- Запуск пользовательского Python-кода в изолированном Docker-контейнере.
- Автопроверка решений через pytest.
- Чат с ИИ-тьютором.
- Адаптация подсказок по SRL-фазам и состоянию студента.
- Сохранение состояния учебной сессии.
- Метрики прогресса, подсказок, эмоций и текущей фазы обучения.

## Архитектура проекта

```text
.
├── Dockerfile              # образ backend-приложения
├── Dockerfile.sandbox      # минимальный образ для запуска пользовательского кода
├── docker-compose.yml      # запуск backend, docker-proxy и сборка sandbox
├── requirements.txt        # зависимости приложения
├── requirements-dev.txt    # зависимости для локальной разработки и тестов
├── pytest.ini              # настройки pytest
├── src/
│   ├── test_server.py      # FastAPI backend и основные API endpoint-ы
│   ├── app/                # SRL, moral schemas, LLM, sandbox, метрики
│   └── ui_moral/           # HTML/CSS/JS интерфейс
├── tests/                  # unit/regression-тесты
├── tutor-data/
│   ├── tasks/              # базовые и сгенерированные задания
│   └── sessions/           # runtime-состояния сессий
└── logs/                   # runtime-логи приложения
```

Папки `logs/`, `tutor-data/sessions/` и `tutor-data/tasks/gen_*/` создаются во время работы приложения и не должны попадать в публичный репозиторий.

## Требования

Для запуска проекта нужны:

- Docker Engine или Docker Desktop.
- Docker Compose v2.
- Современный браузер.
- API-ключ DeepSeek.

Проверить Docker можно так:

```bash
docker --version
docker compose version
```

API-ключ вводится пользователем в интерфейсе перед началом занятия. В серверное хранилище сессий ключ не записывается.

## Быстрый запуск через Docker

1. Склонируйте репозиторий:

```bash
git clone https://github.com/vvvvgross/python-tutor.git
cd python-tutor
```

2. Соберите sandbox-образ:

```bash
docker compose --profile build-only build sandbox-image
```

3. Соберите backend-приложение:

```bash
docker compose build tutor-app
```

4. Запустите приложение:

```bash
docker compose up -d docker-proxy tutor-app
```

5. Проверьте, что backend отвечает:

```bash
curl http://127.0.0.1:8000/api/health
```

Ожидаемый ответ:

```json
{"status":"ok"}
```

6. Откройте приложение в браузере:

```text
http://127.0.0.1:8000
```

## Как пользоваться

1. Откройте главную страницу.
2. Выберите API-провайдера: DeepSeek.
3. Введите API-ключ.
4. Напишите учебную цель, например: `Научиться решать задачи на бинарный поиск`.
5. Выберите уровень подготовки.
6. Запустите генерацию учебной сессии.
7. Последовательно проходите теорию, разбор примера, практическое задание и рефлексию.
8. Для проверки решения используйте кнопку `Test`.

Для перехода дальше достаточно пройти минимум 60% автотестов. Полностью выполненный шаг отмечается зелёным, частично зачтённый шаг отмечается отдельным промежуточным статусом.

## Основные страницы

| URL | Назначение |
|---|---|
| `/` | постановка учебной цели |
| `/workspace?session_id=...` | рабочее пространство занятия |
| `/analytics?session_id=...#token=...` | аналитика текущей сессии |
| `/guide` | справочная страница |

## Основные API

Часть endpoint-ов защищена токеном сессии. Frontend получает этот токен при создании сессии и передаёт его в заголовке `X-Session-Token` или в WebSocket query-параметре.

| Метод | URL | Назначение |
|---|---|---|
| `GET` | `/api/health` | проверка состояния backend |
| `POST` | `/api/validate-key` | проверка API-ключа |
| `POST` | `/api/goal/build` | генерация учебного плана |
| `POST` | `/api/session/create` | создание учебной сессии |
| `POST` | `/api/session/{id}/resume` | восстановление сессии |
| `GET` | `/api/session/{id}` | получение состояния сессии |
| `PATCH` | `/api/session/{id}/state` | сохранение UI-состояния |
| `POST` | `/api/task/generate` | генерация теории или задания |
| `GET` | `/api/task/{task_id}` | получение условия и стартового кода |
| `POST` | `/api/run` | запуск кода пользователя |
| `POST` | `/api/autograde` | запуск pytest-автопроверки |
| `POST` | `/api/theory/expand` | расширение теории |
| `POST` | `/api/session/{id}/reflect` | завершение и рефлексия |
| `GET` | `/api/metrics/{id}` | метрики сессии |
| `GET` | `/api/affective/{id}` | аффективное состояние |
| `WS` | `/ws/tutor/{id}?token=...` | чат с тьютором |

## Docker-сервисы

В `docker-compose.yml` описаны три сервиса:

| Сервис | Назначение |
|---|---|
| `tutor-app` | FastAPI backend и раздача frontend-файлов |
| `docker-proxy` | ограниченный proxy к Docker API |
| `sandbox-image` | build-only сервис для образа `tutor-sandbox:latest` |

`sandbox-image` не запускается как постоянный сервис. Он нужен, чтобы собрать образ, из которого backend создаёт временные sandbox-контейнеры для пользовательского кода.

## Безопасность sandbox

Пользовательский код выполняется в отдельном контейнере с ограничениями:

- сеть отключена;
- root filesystem read-only;
- сброшены Linux capabilities;
- включён `no-new-privileges`;
- ограничены CPU, RAM и количество процессов;
- рабочие временные директории создаются через tmpfs;
- контейнер удаляется после периода неактивности.

Backend не монтирует Docker socket напрямую. Для операций с контейнерами используется `docker-socket-proxy` с ограниченным набором разрешений.

## Остановка и перезапуск

Остановить приложение:

```bash
docker compose down
```

Перезапустить без пересборки:

```bash
docker compose up -d --force-recreate docker-proxy tutor-app
```

Полностью пересобрать после изменения Dockerfile-ов или зависимостей:

```bash
docker compose --profile build-only build sandbox-image
docker compose build tutor-app
docker compose up -d --force-recreate docker-proxy tutor-app
```

Посмотреть логи:

```bash
docker compose logs -f tutor-app docker-proxy
```

## Локальная разработка без Docker

Для полноценного запуска всё равно нужен Docker, потому что пользовательский код и автотесты выполняются в sandbox-контейнерах. Но часть backend-логики и тесты можно запускать локально.

1. Создайте виртуальное окружение:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Установите зависимости:

```bash
pip install -r requirements-dev.txt
```

3. Запустите статическую проверку компиляции:

```bash
python3 -m compileall -q src tests
```

4. Запустите тесты:

```bash
python3 -m pytest -q
```

Если тесты падают с ошибкой `ModuleNotFoundError: No module named 'docker'`, значит зависимости разработки не установлены в активное окружение.

## Конфигурация API-ключей

Основной сценарий: пользователь вводит API-ключ в интерфейсе. Это удобнее и безопаснее для учебного проекта, потому что ключ не нужно хранить в репозитории.

Для локальных экспериментов можно создать файл `src/config.yaml`, но его нельзя публиковать. Проще всего скопировать безопасный пример:

```bash
cp src/config.example.yaml src/config.yaml
```

Формат файла:

```yaml
auth:
  url: https://api.deepseek.com
  token: ""

openai:
  api_key: ""

run_mode: local

running:
  local:
    host: "0.0.0.0"
    port: 8000
  server:
    host: "0.0.0.0"
    port: 8000
```

Файл `src/config.yaml` добавлен в `.gitignore`.

## Runtime-данные

Во время работы приложение создаёт:

- `logs/` — журналы backend-а и тьютора;
- `logs/data_repo/` — NDJSON-события SRL, решений, профилей и LLM-использования;
- `tutor-data/sessions/` — сохранённые состояния учебных сессий;
- `tutor-data/tasks/gen_*/` — LLM-сгенерированные задания и тесты;
- `.pytest_cache/` и `__pycache__/` — локальные кэши Python.

Эти файлы не нужны в GitHub-репозитории и игнорируются.

## Частые проблемы

### `ModuleNotFoundError: No module named 'exceptiongroup'` внутри sandbox

Пересоберите sandbox-образ:

```bash
docker compose --profile build-only build --no-cache sandbox-image
docker compose up -d --force-recreate tutor-app
```

### `/api/health` не отвечает

Проверьте контейнеры:

```bash
docker compose ps
docker compose logs --tail=100 tutor-app docker-proxy
```

### Автопроверка не запускается

Убедитесь, что:

1. собран `tutor-sandbox:latest`;
2. запущен `docker-proxy`;
3. backend видит Docker через `DOCKER_HOST=tcp://docker-proxy:2375`;
4. в `docker-compose.yml` у `docker-proxy` разрешены операции, необходимые для создания и выполнения sandbox-контейнеров.

### LLM долго не отвечает

Иногда DeepSeek отвечает дольше обычного. Интерфейс показывает уведомление, что система ожидает API-ответ и не зависла.
