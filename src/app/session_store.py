"""Durable storage for resumable learning sessions.

The storage contains only educational state and a hash of the opaque access token.
Provider API keys are deliberately never written to disk.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional


class PersistentSessionStore:
    def __init__(self, base_dir: Optional[str] = None) -> None:
        root = base_dir or os.environ.get("SESSION_STORE_DIR") or "/app/tutor-data/sessions"
        if not os.path.isdir("/app") and root.startswith("/app/"):
            root = os.path.join(Path(__file__).resolve().parents[2], "tutor-data", "sessions")
        self.base_dir = Path(root)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _path(self, session_id: str) -> Path:
        safe = "".join(ch for ch in session_id if ch.isalnum() or ch in "_-" )
        if safe != session_id or not safe:
            raise ValueError("Invalid session id")
        return self.base_dir / f"{safe}.json"

    def save(self, session_id: str, data: Dict[str, Any]) -> None:
        payload = dict(data)
        payload["session_id"] = session_id
        payload["updated_at"] = time.time()
        target = self._path(session_id)
        with self._lock:
            fd, tmp_name = tempfile.mkstemp(prefix=f".{session_id}.", suffix=".tmp", dir=self.base_dir)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, ensure_ascii=False, indent=2)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp_name, target)
            finally:
                if os.path.exists(tmp_name):
                    os.remove(tmp_name)

    def load(self, session_id: str) -> Optional[Dict[str, Any]]:
        path = self._path(session_id)
        if not path.exists():
            return None
        with self._lock:
            try:
                with path.open("r", encoding="utf-8") as fh:
                    return json.load(fh)
            except (OSError, json.JSONDecodeError):
                return None

    def delete(self, session_id: str) -> None:
        with self._lock:
            try:
                self._path(session_id).unlink(missing_ok=True)
            except OSError:
                pass
