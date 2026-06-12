import json
import os
import time
import threading
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

_LOG_MAX_BYTES = 50 * 1024 * 1024  # 50 MB per log file
_logger = logging.getLogger(__name__)


class DataRepository:
    """
    Простое файловое хранилище на основе NDJSON (одна JSON‑запись на строку).

    Структура директорий по умолчанию (внутри BASE_DIR, см. _resolve_base_dir):
        /sessions
        /tasks
        /srl
        /affective
        /decisions
        /profiles
        /llm_usage
        /ab_tests

    Каждый тип события пишет в свой *.ndjson файл:
        /<bucket>/<event_type>.ndjson

    Хранилище специально сделано максимально простым и безопасным:
    - Только добавление строк (append‑only)
    - Последовательная запись под блокировкой процесса
    - Ротация файлов при достижении лимита размера
    """

    DEFAULT_BUCKETS = {
        "session": "sessions",
        "task": "tasks",
        "srl": "srl",
        "affective": "affective",
        "decision": "decisions",
        "profile": "profiles",
        "llm_usage": "llm_usage",
        "ab_test": "ab_tests",
    }

    def __init__(self, base_dir: Optional[str] = None) -> None:
        self.base_dir = self._resolve_base_dir(base_dir)
        self._write_lock = threading.RLock()
        self._ensure_structure()

    @staticmethod
    def _resolve_base_dir(base_dir: Optional[str]) -> Path:
        """
        Определяет базовую директорию:
        - ENV DATA_REPO_DIR, если задана
        - Иначе /app/data_repo (в Docker-контейнере)
        - Локально при запуске вне Docker можно переопределить через параметр
        """
        if base_dir:
            return Path(base_dir).absolute()

        env_dir = os.environ.get("DATA_REPO_DIR")
        if env_dir:
            return Path(env_dir).absolute()

        # Базовый вариант по умолчанию для контейнера
        return Path("/app/data_repo").absolute()

    def _ensure_structure(self) -> None:
        """Создает базовую структуру директорий, если её ещё нет."""
        for bucket in self.DEFAULT_BUCKETS.values():
            (self.base_dir / bucket).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def log_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        """
        Записывает событие в соответствующий bucket.

        :param event_type: один из ключей DEFAULT_BUCKETS (session, task, srl, ...)
        :param payload: произвольный словарь; будет дополнен полями:
            - event_type
            - ts (unix timestamp)
        """
        bucket = self.DEFAULT_BUCKETS.get(event_type)
        if not bucket:
            # Для прототипа — не падаем, а просто игнорируем неизвестный тип
            return

        record = dict(payload or {})
        record.setdefault("event_type", event_type)
        record.setdefault("ts", time.time())

        file_path = self.base_dir / bucket / f"{event_type}.ndjson"
        try:
            with self._write_lock:
                if file_path.exists() and file_path.stat().st_size >= _LOG_MAX_BYTES:
                    rotated = file_path.with_suffix(".ndjson.1")
                    if rotated.exists():
                        rotated.unlink()
                    file_path.rename(rotated)
                with file_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
        except Exception as exc:
            _logger.warning("Cannot persist %s event: %s", event_type, exc)

    def _iter_events(
        self,
        event_type: str,
        bucket_override: Optional[str] = None,
        filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ):
        """Итератор по событиям заданного типа с ленивой фильтрацией."""
        bucket = bucket_override or self.DEFAULT_BUCKETS.get(event_type)
        if not bucket:
            return

        file_path = self.base_dir / bucket / f"{event_type}.ndjson"
        if not file_path.exists():
            return

        try:
            with file_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if filter_fn is None or filter_fn(obj):
                        yield obj
        except Exception:
            return

    def load_session_events(self, session_id: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Загружает все события по session_id из всех стандартных бакетов.
        Это удобный метод для офлайнового анализа одной учебной сессии.
        """
        result: Dict[str, List[Dict[str, Any]]] = {
            key: [] for key in self.DEFAULT_BUCKETS.keys()
        }

        def _by_session(ev: Dict[str, Any]) -> bool:
            return ev.get("session_id") == session_id

        for event_type in self.DEFAULT_BUCKETS.keys():
            result[event_type] = list(self._iter_events(event_type, filter_fn=_by_session))  # type: ignore[arg-type]

        return result

    def query_events(
        self,
        event_type: str,
        filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Универсальный метод выборки событий по типу с произвольным фильтром.
        Используется для простых исследовательских запросов и метрик.
        """
        items: List[Dict[str, Any]] = []
        for ev in self._iter_events(event_type, filter_fn=filter_fn):
            items.append(ev)
            if limit is not None and len(items) >= limit:
                break
        return items


# Глобальный синглтон по умолчанию, чтобы не плодить экземпляры во всём приложении
_global_repo: Optional[DataRepository] = None


def get_data_repository() -> DataRepository:
    """Ленивая инициализация глобального репозитория."""
    global _global_repo
    if _global_repo is None:
        _global_repo = DataRepository()
    return _global_repo


