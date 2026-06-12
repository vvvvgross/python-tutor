import ast
import asyncio
import base64
import json
import os
import re
import secrets
import shutil
import hashlib
import hmac
import textwrap
import uuid
from typing import Any, Dict, Optional

import docker
import numpy as np
import time
from fastapi import FastAPI, HTTPException, WebSocketDisconnect, WebSocket, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.terminal_handler import TerminalHandler
from app.virtual_tutor import DummyVirtualTutor, VirtualTutor
from app.moral_engine import MoralProfileEngine
from app.data_repository import get_data_repository
from app.aem_metric import get_aem_calculator
from app.srl_analyzer import TaskInteraction, interaction_to_dict
from json_repair import repair_json
from app.oai_interface import Interface
from app.session_store import PersistentSessionStore

app = FastAPI()

# ─── New API models ───────────────────────────────────────────────────────────

class GoalBuildRequest(BaseModel):
    goal_text:     str
    language:      str = "python"
    level:         str = "beginner"    # beginner | intermediate | advanced
    duration_min:  int = 45
    preference:    str = "balanced"   # practice | balanced | theory
    api_key:       str = ""
    api_provider:  str = "deepseek"   # deepseek | openai
    self_efficacy: int = 3            # Zimmermann SRL Forethought: self-efficacy belief (1-5)

class SessionCreateRequest(BaseModel):
    goal_id:       str
    block_id:      str
    user_prefs:    dict = Field(default_factory=dict)
    block:         Optional[dict] = None   # полный блок из /api/goal/build
    goal:          Optional[dict] = None   # цель из /api/goal/build
    api_key:       str = ""
    api_provider:  str = "deepseek"
    self_efficacy: int = 3   # Zimmermann SRL Forethought: self-efficacy belief (1-5)

class AutogradeRequest(BaseModel):
    session_id: str
    task_id:    str
    files:      dict = Field(default_factory=dict)  # {"starter_code.py": "..."} — optional

class TaskGenerateRequest(BaseModel):
    session_id:      str
    step_index:      int
    step_type:       str = "task"    # micro_lesson | exercise | task | reflection
    step_title:      str
    step_content_md: str = ""
    goal_title:      str
    level:           str = "beginner"

class TheoryExpandRequest(BaseModel):
    session_id:   str
    goal_title:   str
    base_topic:   str
    user_request: str

class RunRequest(BaseModel):
    session_id: str
    task_id:    str
    code:       str

class ReflectionRequest(BaseModel):
    self_rating: int   # 1-5 (self-judgment)
    what_worked: str   # self-satisfaction
    difficulty:  str   # causal attribution
    next_time:   str   # adaptive inference

class SessionResumeRequest(BaseModel):
    session_token: str
    api_key: str = ""
    api_provider: str = "deepseek"

class SessionStatePatch(BaseModel):
    # Only transient UI data may be restored from the browser.
    # Test metrics and reflection results are written only by server endpoints.
    editor_code: Optional[Any] = None    # accepts dict {task_id: code} or str (old frontend format)
    theory_cache: Optional[dict] = None

# ─── Container constants ──────────────────────────────────────────────────────
SANDBOX_IMAGE   = os.environ.get("SANDBOX_IMAGE", "tutor-sandbox:latest")
SANDBOX_PREFIX  = "tutor_sess_"
SANDBOX_TIMEOUT = 15 * 60  # 15 минут неактивности → удалить контейнер
MAX_ACTIVE_SESSIONS = 10   # жёсткий лимит одновременных сессий

# ─── DataRepository path: ensure logs go to project/logs/data_repo ───────────
if not os.environ.get('DATA_REPO_DIR'):
    _repo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs', 'data_repo')
    os.makedirs(_repo_path, exist_ok=True)
    os.environ['DATA_REPO_DIR'] = os.path.abspath(_repo_path)

# ─── In-memory goal session store ────────────────────────────────────────────
# { session_id -> { goal, block, task_id, current_step, current_step_type, srl_tracking } }
goal_sessions: Dict[str, dict] = {}
session_store = PersistentSessionStore()

# Shared LLM interface for goal building (fallback / health checks)
_interface = Interface()

# ─── Per-provider helpers ─────────────────────────────────────────────────────
PROVIDER_BASE_URLS = {
    "deepseek": "https://api.deepseek.com",
    "openai":   "https://api.openai.com/v1",
}
PROVIDER_TEST_MODELS = {
    "deepseek": "deepseek-chat",
    "openai":   "gpt-4o-mini",
}

def _make_interface(api_key: str, provider: str = "deepseek") -> Interface:
    base_url = PROVIDER_BASE_URLS.get(provider, PROVIDER_BASE_URLS["deepseek"])
    return Interface(token=api_key, base_url=base_url, model_name=_model_for_provider(provider))

def _get_session_interface(session_id: str) -> Interface:
    """Возвращает Interface с ключом пользователя. Берётся из tutor, api_key в goal_sessions не хранится."""
    if session_id not in goal_sessions:
        raise HTTPException(
            status_code=404,
            detail="Сессия не найдена (истекла или удалена). Вернитесь на главную и начните заново."
        )
    tutor = session_manager.tutors.get(session_id)
    iface = getattr(tutor, "interface", None) if tutor else None
    if not iface:
        raise HTTPException(status_code=401, detail="API key not set for session. Please start a new session.")
    return iface

def _model_for_provider(provider: str) -> str:
    return PROVIDER_TEST_MODELS.get(provider, PROVIDER_TEST_MODELS["deepseek"])

def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()

def _public_session_state(data: dict) -> dict:
    return {k: v for k, v in data.items() if k not in {"token_hash"}}

def _persist_session(session_id: str) -> None:
    data = goal_sessions.get(session_id)
    if not data:
        return
    snapshot = dict(data)
    tutor = session_manager.get_tutor(session_id) if "session_manager" in globals() else None
    if tutor:
        snapshot["metrics"] = dict(getattr(tutor, "metrics", {}) or {})
        snapshot["messages"] = list(getattr(tutor, "messages", []) or [])
        snapshot["task_interactions"] = [interaction_to_dict(i) for i in getattr(tutor, "task_interactions", []) or []]
    session_store.save(session_id, snapshot)

def _restore_session(session_id: str, api_key: str = "", api_provider: str = "deepseek") -> Optional[dict]:
    if session_id in goal_sessions:
        return goal_sessions[session_id]
    saved = session_store.load(session_id)
    if not saved:
        return None
    goal_sessions[session_id] = saved
    tutor = VirtualTutor(session_id, moral_profile_engine=session_manager.moral_profile_engine)
    tutor.metrics.update(saved.get("metrics") or {})
    if saved.get("messages"):
        tutor.messages = saved["messages"]
    try:
        from app.srl_analyzer import interaction_from_dict
        tutor.task_interactions = [interaction_from_dict(x) for x in saved.get("task_interactions", [])]
    except Exception:
        tutor.task_interactions = []
    if api_key:
        tutor.interface = _make_interface(api_key, api_provider)
        _propagate_interface(tutor, tutor.interface)
    tutor.set_goal_context(saved)
    session_manager.tutors[session_id] = tutor
    return saved

def _require_session_token(session_id: str, token: Optional[str]) -> dict:
    data = goal_sessions.get(session_id) or _restore_session(session_id)
    if not data:
        raise HTTPException(status_code=404, detail="Session not found")
    expected = data.get("token_hash")
    if not token or not expected or not hmac.compare_digest(_token_hash(token), expected):
        raise HTTPException(status_code=401, detail="Invalid session token")
    return data

def _ws_authorized(session_id: str, websocket: WebSocket) -> bool:
    token = websocket.query_params.get("token", "")
    try:
        _require_session_token(session_id, token)
        return True
    except HTTPException:
        return False

def _propagate_interface(tutor, iface) -> None:
    """Propagate user's API interface to all tutor sub-components."""
    try:
        de = getattr(tutor, 'decision_engine', None)
        if de:
            am = getattr(de, 'affective_monitor', None)
            if am:
                am.interface = iface
            lo = getattr(de, 'llm_orchestrator', None)
            if lo:
                lo.interface = iface
                active_provider = "openai" if "openai.com" in getattr(iface, "base_url", "") else "deepseek"
                for descriptor in getattr(lo, "models", {}).values():
                    descriptor.provider = active_provider
                    descriptor.model_name = getattr(iface, "model_name", descriptor.model_name)
        for ms in getattr(tutor, 'ms_list', []):
            ms.oai_interface = iface
    except Exception as _e:
        print(f"⚠️ _propagate_interface error (non-critical): {_e}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://virtual-tutor.ru",
        "https://www.virtual-tutor.ru",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Session-Token"],
)

_UI_MORAL_PATH = os.path.join(os.path.dirname(__file__), "ui_moral")

def _serve_ui(filename: str):
    from fastapi.responses import FileResponse
    path = os.path.join(_UI_MORAL_PATH, filename)
    if os.path.exists(path):
        return FileResponse(path)
    raise HTTPException(status_code=404, detail=f"{filename} not found")

# ─── Health check ────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health_check():
    return {"status": "ok"}


class ValidateKeyRequest(BaseModel):
    api_key:  str
    provider: str = "deepseek"   # deepseek | openai

@app.post("/api/validate-key")
async def validate_api_key(req: ValidateKeyRequest):
    """Проверяет работоспособность API-ключа пользователя."""
    if not req.api_key.strip():
        return {"valid": False, "error": "Ключ не может быть пустым"}
    model = PROVIDER_TEST_MODELS.get(req.provider, "deepseek-chat")
    iface = _make_interface(req.api_key.strip(), req.provider)
    try:
        result = await asyncio.to_thread(
            iface._make_api_request,
            "/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 5},
        )
        if result and "choices" in result:
            return {"valid": True, "provider": req.provider, "model": model}
        return {"valid": False, "error": "Ключ не принят API — проверь правильность ключа"}
    except Exception as e:
        return {"valid": False, "error": str(e)}


# ─── Page routes ─────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """New entry point: Goal Setup page."""
    return _serve_ui("goal.html")

@app.get("/workspace")
async def workspace_page():
    """Learning Studio page."""
    return _serve_ui("workspace.html")

@app.get("/legacy")
async def legacy_page():
    raise HTTPException(status_code=410, detail="Legacy UI disabled for security reasons")

@app.get("/ui_moral/")
async def ui_moral_root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/")

@app.get("/ui_moral/{file_path:path}")
async def ui_moral_static(file_path: str):
    """Serve public UI assets only; block legacy files and path traversal."""
    from fastapi.responses import FileResponse
    if file_path in {"index.html", "script.js", "styles.css"}:
        raise HTTPException(status_code=410, detail="Legacy UI disabled for security reasons")
    allowed_ext = {".html", ".js", ".css", ".ico", ".png", ".svg", ".woff", ".woff2"}
    base = os.path.abspath(_UI_MORAL_PATH)
    full = os.path.abspath(os.path.join(base, file_path))
    _, ext = os.path.splitext(full)
    if not full.startswith(base + os.sep) or ext.lower() not in allowed_ext:
        raise HTTPException(status_code=404, detail="File not found")
    if os.path.isfile(full):
        return FileResponse(full)
    raise HTTPException(status_code=404, detail="File not found")

# Serve goal.html and workspace.html as static (direct file access)
@app.get("/goal.html")
async def goal_html(): return _serve_ui("goal.html")

@app.get("/workspace.html")
async def workspace_html(): return _serve_ui("workspace.html")

@app.get("/analytics")
async def analytics_page():
    """Session analytics report page."""
    return _serve_ui("analytics.html")

@app.get("/guide")
async def guide_page():
    """User guide page."""
    return _serve_ui("guide.html")

@app.get("/guide.html")
async def guide_html_page(): return _serve_ui("guide.html")

@app.get("/{filename}")
async def serve_static_file(filename: str):
    """Serve any static file from ui_moral directory (JS, CSS, images)."""
    # Only serve known static file types
    allowed_ext = {'.js', '.css', '.ico', '.png', '.svg', '.woff', '.woff2'}
    _, ext = os.path.splitext(filename)
    if ext.lower() in allowed_ext:
        if filename in {"script.js", "styles.css"}:
            raise HTTPException(status_code=410, detail="Legacy UI disabled for security reasons")
        from fastapi.responses import FileResponse
        full = os.path.join(_UI_MORAL_PATH, filename)
        if os.path.exists(full) and os.path.isfile(full):
            return FileResponse(full)
    raise HTTPException(status_code=404, detail=f"{filename} not found")

# Block ui_dummy
@app.get("/ui_dummy/{path:path}")
async def ui_dummy_blocked(path: str):
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/")

@app.get("/ui_dummy/")
async def ui_dummy_root_blocked():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/")

class SessionManager:
    def __init__(self):
        self.tutors: Dict[str, VirtualTutor] = {}
        self.dummy_tutors: Dict[str, DummyVirtualTutor] = {}
        self.active_connections: Dict[str, WebSocket] = {}
        self.creating_sessions: set = set()  # Множество ID сессий, которые сейчас создаются
        self.session_containers: Dict[str, Any] = {}    # session_id → Container
        self.session_last_active: Dict[str, float] = {} # session_id → timestamp
        try:
            self.docker_client = docker.from_env()
            self.docker_client.ping()
            print("✅ Docker connection established")
        except Exception as e:
            print(f"⚠️ Docker connection error: {str(e)}")
            print("Running in limited mode without Docker integration")
            self.docker_client = None
        # Компоненты для анализа профилей и данных сессий
        self.moral_profile_engine = MoralProfileEngine()
        self.data_repo = get_data_repository()

    def get_container_name(self, session_id: str) -> str:
        return f"{SANDBOX_PREFIX}{session_id}"

    async def start_container(self, session_id: str):
        """Создать изолированный контейнер для сессии."""
        if not self.docker_client:
            return None
        name = self.get_container_name(session_id)
        # Удалить старый контейнер с таким именем, если есть (после краша сервера)
        try:
            old = await asyncio.to_thread(self.docker_client.containers.get, name)
            await asyncio.to_thread(old.remove, force=True)
        except docker.errors.NotFound:
            pass
        except Exception:
            pass
        container = await asyncio.to_thread(
            self.docker_client.containers.run,
            SANDBOX_IMAGE,
            name=name,
            detach=True,
            tty=True,
            command="tail -f /dev/null",
            environment={"PYTHONUNBUFFERED": "1", "LANG": "C.UTF-8"},
            # ── Ресурсные лимиты на пользовательский контейнер ──
            # VM: 2 vCPU / 4 GB RAM / гарантия CPU 10% (= 0.2 vCPU гарантировано, burst до 2)
            # Замер от 2026-03-28: idle ~2.7 MiB, пик pytest ~14 MiB, CPU-пик 15% за < 0.1 сек
            mem_limit="200m",     # 10 контейнеров × 200 MB = 2 GB / 4 GB
            memswap_limit="200m", # swap отключён (mem == memswap)
            cpu_period=100000,
            cpu_quota=15000,      # 15% одного vCPU на контейнер; pytest завершается за ~1-2 с
                                  # 10 контейнеров × 15% = 150% — равномерно по обоим ядрам ВМ
            network_disabled=True,
            read_only=True,
            cap_drop=["ALL"],
            security_opt=["no-new-privileges"],
            pids_limit=64,
            user="10001:10001",
            # mode=1777 is REQUIRED: the root filesystem is read-only and the
            # container runs as uid 10001. A tmpfs without an explicit mode is
            # created root:root 0755 (kernel default 0777 & umask), so the
            # sandbox user cannot create /app/tasks/<task_id> — put_archive then
            # fails with HTTP 404 ("Could not find the file ... in container").
            # Making the writable workspace world-writable with the sticky bit
            # (like /tmp) lets each session's container drop in its task files.
            tmpfs={
                "/tmp": "rw,noexec,nosuid,size=64m,mode=1777",
                "/app/tasks": "rw,noexec,nosuid,size=128m,mode=1777",
            },
        )
        self.session_containers[session_id] = container
        self.session_last_active[session_id] = time.time()
        health = await asyncio.to_thread(
            container.exec_run,
            [
                "python3",
                "-c",
                "import sys, pytest; import exceptiongroup; "
                "print(f'python={sys.version_info.major}.{sys.version_info.minor} pytest={pytest.__version__}')",
            ],
            demux=True,
            workdir="/app",
        )
        if health.exit_code != 0:
            stderr = (health.output[1] or health.output[0] or b"").decode(errors="replace").strip()
            await asyncio.to_thread(container.remove, force=True)
            self.session_containers.pop(session_id, None)
            self.session_last_active.pop(session_id, None)
            raise RuntimeError(
                "Sandbox image is missing pytest runtime dependencies. "
                "Rebuild tutor-sandbox:latest with `docker compose --profile build-only build --no-cache sandbox-image`. "
                f"Container check failed: {stderr or 'no output'}"
            )
        print(f"🐳 Container started: {name}")
        return container

    async def stop_container(self, session_id: str):
        """Уничтожить контейнер сессии."""
        container = self.session_containers.pop(session_id, None)
        self.session_last_active.pop(session_id, None)
        if container:
            try:
                await asyncio.to_thread(container.remove, force=True)
                print(f"🗑️ Container removed: {self.get_container_name(session_id)}")
            except Exception:
                pass

    async def _reaper_loop(self):
        """Фоновая задача: удалять контейнеры неактивных сессий."""
        while True:
            await asyncio.sleep(60)
            try:
                now = time.time()
                expired = [
                    sid for sid, ts in list(self.session_last_active.items())
                    if now - ts > SANDBOX_TIMEOUT
                ]
                for sid in expired:
                    print(f"♻️ Reaper: удаляю контейнер сессии {sid}")
                    await self.stop_container(sid)
                    # Состояние сессии сохраняется на диске и может быть восстановлено после перезагрузки страницы.
                    _persist_session(sid)
                    goal_sessions.pop(sid, None)
                    self.tutors.pop(sid, None)
            except Exception as exc:
                print(f"⚠️ Reaper error (loop continues): {exc}")

    def get_or_create_tutor(self, user_id: str, use_moral: bool = True):
        """Возвращает существующего тьютора или создаёт нового."""
        # ВСЕГДА используем use_moral=True, dummy сессии отключены
        use_moral = True
        if user_id not in self.tutors:
            self.tutors[user_id] = VirtualTutor(user_id, moral_profile_engine=self.moral_profile_engine)
        return self.tutors[user_id]

    def get_tutor(self, user_id: str):
        """Возвращает существующего тьютора или None."""
        if user_id in self.tutors:
            return self.tutors[user_id]
        elif user_id in self.dummy_tutors:
            return self.dummy_tutors[user_id]
        return None

    async def create_session(self, user_id: str, use_moral_schemes: bool = True):
        # ВСЕГДА создаем только moral сессии, dummy сессии отключены
        if user_id not in self.tutors:
            self.tutors[user_id] = VirtualTutor(user_id, moral_profile_engine=self.moral_profile_engine)
        return {"status": "created", "type": "moral"}

    async def cleanup_session(self, user_id: str):
        """
        Очистка сессии и запись итоговых A/B‑метрик (рост TMR / снижение FS).
        """
        # Перед удалением пытаемся зафиксировать агрегированные результаты
        try:
            from app.srl_analyzer import SRLAnalyzer, interaction_from_dict

            # Загружаем все события по сессии
            all_events = self.data_repo.load_session_events(user_id)

            # Восстанавливаем SRL‑интеракции
            interactions = [
                interaction_from_dict(ev)
                for ev in all_events.get("task", [])
            ]
            if interactions:
                srl_analyzer = SRLAnalyzer()
                srl_state = srl_analyzer.get_srl_state(user_id, interactions)
            else:
                srl_state = None

            # Фрустрацию берём как среднее по logged affective
            affective_events = all_events.get("affective", [])
            if affective_events:
                fs_vals = [
                    float(ev.get("state", {}).get("frustration_score", 0.0))
                    for ev in affective_events
                ]
                avg_fs = sum(fs_vals) / max(1, len(fs_vals))
            else:
                avg_fs = 0.0

            # Вытаскиваем последний лог профиля (если был)
            profiles = all_events.get("profile", [])
            profile_snapshot = profiles[-1]["profile"] if profiles else {}

            # Логируем итог сессии в ab_tests как метрику результата A/B
            self.data_repo.log_event(
                "ab_test",
                {
                    "session_id": user_id,
                    "final_srl": srl_state,
                    "avg_frustration": avg_fs,
                    "final_profile": profile_snapshot,
                },
            )
        except Exception:
            # Не мешаем основному потоку удаления сессии
            pass

        if user_id in self.tutors:
            del self.tutors[user_id]
        if user_id in self.dummy_tutors:
            del self.dummy_tutors[user_id]

    async def execute_in_container(self, session_id: str, command: str):
        if not self.docker_client:
            return "Docker integration is not available"
        handler = TerminalHandler(self.docker_client, self.get_container_name(session_id))
        return await handler.execute(command)

session_manager = SessionManager()

@app.on_event("startup")
async def on_startup():
    asyncio.create_task(session_manager._reaper_loop())

@app.on_event("shutdown")
async def on_shutdown():
    for sid in list(session_manager.session_last_active.keys()):
        try:
            await session_manager.stop_container(sid)
        except Exception:
            pass

# ═════════════════════════════════════════════════════════════
# NEW ENDPOINTS: Goal Build, Session Create, Autograde, Run
# ═════════════════════════════════════════════════════════════

# ─── Helper: получить TerminalHandler для сессии ─────────────
def _get_handler(session_id: str) -> TerminalHandler:
    if not session_manager.docker_client:
        raise HTTPException(status_code=503, detail="Среда выполнения кода недоступна: Docker API не подключён.")
    container_name = session_manager.get_container_name(session_id)
    try:
        session_manager.docker_client.containers.get(container_name)
    except docker.errors.NotFound:
        raise HTTPException(
            status_code=404,
            detail="Сессия истекла — контейнер удалён (15 мин неактивности). Вернитесь на главную и начните заново."
        )
    return TerminalHandler(session_manager.docker_client, container_name)

# ─── Helper: strip function bodies so starter_code is always a skeleton ─────
def _strip_function_bodies(source: str) -> str:
    """
    Replaces the body of every top-level function with a docstring (if present)
    + `pass`, so the LLM cannot sneak a complete solution into starter_code.
    The `if __name__ == '__main__':` block is left untouched.
    Falls back to returning the original source if parsing fails.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    lines = source.splitlines(keepends=True)
    replacements: list[tuple[int, int, str]] = []  # (start_line, end_line, new_body)

    # Strip both module-level functions and class methods. The earlier
    # implementation handled only module-level functions, therefore a task
    # generated as a class could expose ready-made method implementations.
    parent_map = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not isinstance(parent_map.get(node), (ast.Module, ast.ClassDef)):
            continue

        body = node.body
        # Keep existing docstring if present
        docstring = ""
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, (ast.Constant, ast.Str))
        ):
            raw = ast.get_source_segment(source, body[0]) or ""
            docstring = f"    {raw}\n" if raw else ""
            body_start = body[1] if len(body) > 1 else None
        else:
            body_start = body[0] if body else None

        # Determine if body is already just `pass` — skip if so
        non_trivial = [
            n for n in (body[1:] if docstring else body)
            if not (isinstance(n, ast.Pass) or
                    (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)))
        ]
        if not non_trivial:
            continue  # already a skeleton

        indent = "    "
        new_body = docstring + f"{indent}# TODO: реализуй функцию\n{indent}pass\n"
        # line numbers in AST are 1-based
        func_body_start = body[0].lineno
        func_body_end   = node.end_lineno
        replacements.append((func_body_start, func_body_end, new_body))

    if not replacements:
        return source

    # Apply replacements from bottom to top so line numbers stay valid
    replacements.sort(key=lambda x: x[0], reverse=True)
    for start, end, new_body in replacements:
        lines[start - 1 : end] = [new_body]

    return "".join(lines)


# ─── Helper: validate task_id to prevent injection ───────────
_TASK_ID_RE = re.compile(r'^[a-zA-Z0-9_-]+$')

def _validate_task_id(task_id: str) -> None:
    if not _TASK_ID_RE.match(task_id):
        raise HTTPException(status_code=400, detail="Invalid task_id")

# ─── Helper: save code to container via base64 ───────────────
async def _save_code_to_container(handler: TerminalHandler, task_id: str, code: str) -> str:
    """Safely writes code only inside the current generated task directory."""
    _validate_task_id(task_id)
    return await handler.save_file(f"/app/tasks/{task_id}/starter_code.py", code)


# ─── Helper: parse pytest output ─────────────────────────────
def _parse_pytest_output(output: str) -> dict:
    """Parses `pytest -v --tb=short` output into structured result."""
    tests = []
    lines = output.splitlines()

    # Collect test outcomes
    for line in lines:
        if ' PASSED' in line and '::' in line:
            name = line.split('::')[-1].split(' ')[0].strip()
            tests.append({"name": name, "status": "passed", "error": None, "trace": None})
        elif ' FAILED' in line and '::' in line:
            parts = line.split('::')
            name = parts[-1].split()[0].strip()
            error = parts[-1].split(' - ', 1)[1].strip() if ' - ' in parts[-1] else None
            tests.append({"name": name, "status": "failed", "error": error, "trace": None})
        elif ' ERROR' in line and '::' in line:
            name = line.split('::')[-1].split(' ')[0].strip()
            tests.append({"name": name, "status": "error", "error": "Ошибка выполнения теста", "trace": None})

    # Parse FAILURES/ERRORS sections into per-test blocks
    failure_blocks: dict = {}
    in_failures = False
    current_test_name = ""
    current_block_lines: list = []

    for line in lines:
        if re.search(r'=+ (FAILURES|ERRORS) =+', line):
            in_failures = True
            continue
        if in_failures:
            if line.startswith('_') and line.strip('_ '):
                name = line.strip().strip('_').strip()
                if name:
                    if current_test_name and current_block_lines:
                        failure_blocks[current_test_name] = '\n'.join(current_block_lines).strip()
                    current_test_name = name
                    current_block_lines = []
            elif line.startswith('=') and len(line) > 10:
                if current_test_name and current_block_lines:
                    failure_blocks[current_test_name] = '\n'.join(current_block_lines).strip()
                in_failures = False
            else:
                current_block_lines.append(line)

    if current_test_name and current_block_lines:
        failure_blocks[current_test_name] = '\n'.join(current_block_lines).strip()

    print(f"🔍 failure_blocks keys: {list(failure_blocks.keys())}")
    # Attach failure blocks to tests + enrich with beginner-friendly fields
    for test in tests:
        if test["status"] in ("failed", "error"):
            block = failure_blocks.get(test["name"], "")
            if block:
                test["trace"] = block
            _enrich_test_error(test, block)

    passed = sum(1 for t in tests if t["status"] == "passed")
    failed = sum(1 for t in tests if t["status"] in ("failed", "error"))

    # Last summary line
    signals = {}
    for line in reversed(lines):
        if 'AssertionError' in line:
            signals['error_type'] = 'assertion_error'
            break
        if 'RecursionError' in line:
            signals['error_type'] = 'recursion_error'
            break
        if 'NameError' in line or 'AttributeError' in line:
            signals['error_type'] = 'name_error'
            break

    return {
        "summary": {"passed": passed, "failed": failed, "total": passed + failed},
        "tests":   tests,
        "signals": signals,
        "raw_output": output[-2000:] if len(output) > 2000 else output,
    }


def _autograde_passed_enough(summary: dict) -> bool:
    """A task is passable for progression when at least 60% tests pass."""
    total = int((summary or {}).get("total") or 0)
    passed = int((summary or {}).get("passed") or 0)
    if total <= 0:
        return False
    required = (3 * total + 4) // 5  # ceil(total * 0.6)
    return passed >= required


def _strip_py_quotes(s: str) -> str:
    """Remove surrounding Python string quotes from a repr value."""
    s = s.strip()
    for q in ('"""', "'''", '"', "'"):
        if len(s) > 2 * len(q) and s.startswith(q) and s.endswith(q):
            return s[len(q):-len(q)]
    return s


def _enrich_test_error(test: dict, block: str) -> None:
    """Add beginner-friendly fields to a failed/error test in-place."""
    all_text = (block or "") + "\n" + (test.get("error") or "")
    all_lines = all_text.splitlines()

    # ── error_kind ──
    if "AssertionError" in all_text or (
            "assert " in all_text and test["status"] == "failed"):
        test["error_kind"] = "assertion"
    elif "NameError" in all_text or "AttributeError" in all_text:
        test["error_kind"] = "name_error"
    elif "TypeError" in all_text:
        test["error_kind"] = "type_error"
    elif "SyntaxError" in all_text or "IndentationError" in all_text:
        test["error_kind"] = "syntax_error"
    elif "RecursionError" in all_text or "maximum recursion" in all_text:
        test["error_kind"] = "recursion"
    else:
        test["error_kind"] = "other"

    # ── got / expected ──
    # Collect E-lines from block, then the short error field as fallback
    test["got"] = None
    test["expected"] = None

    candidate_lines = []
    for line in all_lines:
        s = line.strip()
        if re.match(r'^E\s+', s):
            candidate_lines.append(re.sub(r'^E\s+', '', s).strip())
    if test.get("error"):
        candidate_lines.append(test["error"].strip())

    for content in candidate_lines:
        content = re.sub(r'^AssertionError:\s*', '', content)
        if not content.startswith('assert '):
            continue
        body = content[len('assert '):]
        if ' == ' not in body:
            continue
        parts = body.split(' == ', 1)
        got_raw = parts[0].strip()
        exp_raw = parts[1].strip()
        # Skip source-code lines where both sides look like bare variable names
        if not re.search(r'["\'\[\]{}\d\s]', got_raw + exp_raw):
            continue
        test["got"]      = _strip_py_quotes(got_raw)[:300]
        test["expected"] = _strip_py_quotes(exp_raw)[:300]
        break

    # ── error_line ──
    test["error_line"] = None
    m = re.search(r'starter_code\.py:(\d+):', all_text)
    if m:
        test["error_line"] = int(m.group(1))

    # ── error_message — first E-line, cleaned ──
    test["error_message"] = None
    for line in all_lines:
        s = line.strip()
        if re.match(r'^E\s+', s):
            test["error_message"] = re.sub(r'^E\s+', '', s).strip()
            break


# ─── Helper: template-based goal block ───────────────────────
def _build_template_block(req: GoalBuildRequest) -> dict:
    """Fallback template block when LLM is unavailable."""
    goal_id  = f"goal_{uuid.uuid4().hex[:8]}"
    block_id = f"block_{uuid.uuid4().hex[:8]}"
    return {
        "goal": {
            "id":               goal_id,
            "title":            req.goal_text[:80],
            "requested_topic":  req.goal_text,
            "language":         req.language,
            "level":            req.level,
            "duration_min":     req.duration_min,
            "success_criteria": "Пройти все автотесты в задании",
        },
        "block": {
            "id":            block_id,
            "outcomes":      [
                f"Понимать основные концепции по теме «{req.goal_text[:40]}»",
                "Уметь применять изученное на практике",
                "Решать задачи по теме",
            ],
            "prerequisites": ["Базовый синтаксис Python"],
            "steps": [
                {"step_id": "s1", "type": "micro_lesson", "title": f"Теория: {req.goal_text[:50]}", "content_md": f"Краткое введение в тему: {req.goal_text[:60]}.", "task_id": None},
                {"step_id": "s2", "type": "exercise",     "title": "Упражнение: разбор примера", "content_md": "Разберём пример по теме.", "task_id": None},
                {"step_id": "s3", "type": "task",         "title": "Практическое задание", "content_md": "Напиши код по теме.", "task_id": None},
                {"step_id": "s4", "type": "reflection",   "title": "Рефлексия", "content_md": "Самооценка (фаза Self-Reflection по Зиммерманну): что удалось, что было трудным и почему, что сделаешь иначе в следующий раз.", "task_id": None},
            ],
        },
    }


# ─── Helper: sanitize LLM JSON output ────────────────────────
def _sanitize_llm_json(s: str) -> str:
    """Sanitize LLM-generated JSON string:
    1. Escape any unescaped control characters (\\n, \\r, \\t, etc.) inside string values.
    2. Remove trailing commas before } or ] (common LLM mistake).
    """
    result = []
    in_string = False
    i = 0
    while i < len(s):
        c = s[i]
        if c == '\\' and in_string:
            # escaped sequence — pass both chars through unchanged
            result.append(c)
            i += 1
            if i < len(s):
                result.append(s[i])
            i += 1
            continue
        if c == '"':
            in_string = not in_string
            result.append(c)
        elif in_string and ord(c) < 0x20:
            # escape any ASCII control character inside a string value
            _esc = {'\n': '\\n', '\r': '\\r', '\t': '\\t', '\b': '\\b', '\f': '\\f'}
            result.append(_esc.get(c, f'\\u{ord(c):04x}'))
        else:
            result.append(c)
        i += 1
    cleaned = ''.join(result)
    # Remove trailing commas before closing } or ]
    cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)
    return cleaned


_RU_STOPWORDS = {"изучить", "научиться", "освоить", "работать", "python", "язык", "основы", "программирование", "для", "при", "про", "как", "что", "это", "тему"}
_RU_ENDINGS = ("иями", "ями", "ами", "ого", "ему", "ыми", "ими", "ение", "ений", "ировать", "ировать", "ать", "ять", "ить", "ых", "ий", "ой", "ая", "ое", "ые", "ам", "ям", "ом", "ем", "ов", "ев", "ы", "и", "а", "я", "у", "ю", "е")

def _topic_stem(word: str) -> str:
    word = word.lower().replace("ё", "е")
    for ending in _RU_ENDINGS:
        if len(word) - len(ending) >= 4 and word.endswith(ending):
            return word[:-len(ending)]
    return word

def _topic_terms(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Zа-яА-ЯёЁ_]{3,}", (text or "").lower())
    return {_topic_stem(w) for w in words if w not in _RU_STOPWORDS}

def _content_matches_goal(goal_text: str, text: str) -> bool:
    terms = _topic_terms(goal_text)
    if not terms:
        return True
    body_terms = _topic_terms(text)
    return bool(terms & body_terms)

def _validate_learning_block(req: GoalBuildRequest, data: dict) -> bool:
    goal = data.get("goal") or {}
    block = data.get("block") or {}
    steps = block.get("steps") or []
    types = [step.get("type") for step in steps]
    if types != ["micro_lesson", "exercise", "task", "reflection"]:
        return False
    combined = " ".join([goal.get("title", "")] + [f"{st.get('title', '')} {st.get('content_md', '')}" for st in steps[:3]])
    return _content_matches_goal(req.goal_text, combined)

def _validate_task_relevance(goal_title: str, statement: str, starter_code: str, level: str) -> tuple[bool, list[str]]:
    """Validate only hard structural failures; relevance/complexity checks remain diagnostic.

    A lexical check cannot reliably reject a valid Russian-language task (e.g.
    synonyms or inflected words).  Earlier code also rejected every advanced
    task because it evaluated a deliberately empty starter-code skeleton.
    """
    errors: list[str] = []
    warnings: list[str] = []
    if not (statement or "").strip():
        errors.append("пустое условие задания")
    if not (starter_code or "").strip():
        errors.append("отсутствует стартовый код")
    if not _content_matches_goal(goal_title, statement):
        warnings.append("в формулировке задания не найдено прямое лексическое совпадение с темой")
    combined = (statement or "").lower()
    if level == "advanced" and not any(marker in combined for marker in ("класс", "метод", "наслед", "рекурс", "генератор", "декоратор", "comprehension")):
        warnings.append("в условии продвинутого задания не выделена сложная конструкция явно")
    return (not errors), errors + warnings


def _count_test_cases(test_code: str) -> int:
    """Number of ``def test_*`` functions in the generated pytest module."""
    return len(re.findall(r'^\s*def\s+test_\w+', test_code or "", re.MULTILINE))


# Minimum complexity each level must satisfy. Deliberately lenient: it only
# rejects tasks that are clearly *too trivial* for the level so the generator
# can retry — it never rejects a task merely for topic wording (that stays a
# warning in _validate_task_relevance), to avoid the over-strict regression.
_DIFFICULTY_FLOORS = {
    "beginner":     {"min_tests": 3, "min_words": 8},
    "intermediate": {"min_tests": 4, "min_words": 12},
    "advanced":     {"min_tests": 4, "min_words": 16},
}

# Lexical markers that show a solution genuinely needs the level's constructs.
_INTERMEDIATE_MARKERS = (
    "список", "спис", "словар", "dict", "list", "строк", "split", "join",
    "append", "цикл", "for ", "while", "if ", "elif", "множеств", "set(",
    "сортир", "филь", "подсчит", "групп",
)
_ADVANCED_MARKERS = (
    "класс", "class ", "метод", "наслед", "рекурс", "recursion", "генератор",
    "yield", "декоратор", "decorator", "comprehension", "стек", "очеред",
    "дерев", "граф", "валидац", "алгоритм",
)


def _assess_task_difficulty(level: str, statement: str, starter_code: str,
                            test_code: str, duration_min: int) -> tuple[bool, list[str]]:
    """Programmatic difficulty gate (problem §3/§5).

    Examines the *generated* task — statement size, number of pytest cases,
    declared functions/classes and the constructs the solution must use — and
    decides whether it is rich enough for the requested level. Returns
    ``(meets_level, reasons)``; ``reasons`` lists why it is considered too easy.
    The caller uses this to regenerate weak tasks instead of showing them.
    """
    s, sc, tc = (statement or ""), (starter_code or ""), (test_code or "")
    floors = _DIFFICULTY_FLOORS.get(level, _DIFFICULTY_FLOORS["beginner"])
    words = len(re.findall(r'\w+', s))
    test_count = _count_test_cases(tc)
    func_count = len(re.findall(r'^\s*def\s+\w+', sc, re.MULTILINE))
    class_count = len(re.findall(r'^\s*class\s+\w+', sc, re.MULTILINE))
    combined = (s + " " + sc + " " + tc).lower()
    reasons: list[str] = []

    if test_count < floors["min_tests"]:
        reasons.append(f"мало тестовых случаев ({test_count} < {floors['min_tests']})")
    if words < floors["min_words"]:
        reasons.append(f"условие слишком короткое ({words} слов)")

    if level == "intermediate":
        if not any(m in combined for m in _INTERMEDIATE_MARKERS):
            reasons.append("нет работы со списками/словарями/строками или ветвлением — слишком примитивно для среднего уровня")
    elif level == "advanced":
        has_struct = class_count >= 1 or func_count >= 2
        if not (has_struct or any(m in combined for m in _ADVANCED_MARKERS)):
            reasons.append("нет ООП, рекурсии, структур данных или нетривиального алгоритма — недостаточно для продвинутого уровня")
        # A long advanced session should not collapse to a one-function toy.
        if duration_min >= 60 and func_count <= 1 and class_count == 0 and test_count < 5:
            reasons.append("для длинной продвинутой сессии задание слишком компактно")

    return (len(reasons) == 0), reasons

# ─── POST /api/goal/build ─────────────────────────────────────
@app.post("/api/goal/build")
async def build_goal(req: GoalBuildRequest):
    """
    Generate a personalised learning block using LLM.
    Falls back to template if LLM fails.
    """
    goal_id  = uuid.uuid4().hex[:8]
    block_id = uuid.uuid4().hex[:8]

    level_map = {"beginner": "новичок", "intermediate": "средний уровень", "advanced": "продвинутый"}
    pref_map  = {"practice": "больше практики", "balanced": "сбалансированно", "theory": "больше теории"}

    prompt = f"""Ты — методист, эксперт по обучению программированию. Создай учебный блок на русском языке.

Цель студента: {req.goal_text}
Язык: {req.language}
Уровень: {level_map.get(req.level, req.level)}
Длительность: {req.duration_min} минут
Предпочтение: {pref_map.get(req.preference, req.preference)}

Структура шагов должна соответствовать циклу SRL Зиммерманна по содержанию (не по названию):
- Шаг micro_lesson: даёт теоретическую базу и ориентировку по теме
- Шаги exercise/task: практика с самоконтролем и применением знаний
- Последний шаг reflection: самооценка — его content_md обозначает три вопроса: что удалось, что было трудным и почему, что сделать иначе

ОБЯЗАТЕЛЬНО ровно 4 шага в этом порядке: micro_lesson → exercise → task → reflection.
Шаг task НЕ пропускать — он нужен как самостоятельное практическое задание после разбора примера.

ВАЖНО: НЕ используй слова "Ориентировка", "Исполнение", "Самооценка" в поле title шагов.
Заголовки шагов ОБЯЗАТЕЛЬНО начинаются строго с: "Теория:", "Разбор примера:", "Практическое задание:", "Рефлексия"
После префикса — конкретное описание содержания шага (НЕ повторять цель студента дословно).
Пример хорошего заголовка: "Теория: классы, объекты и атрибуты в Python", "Разбор примера: создаём класс Animal с наследованием".

Ответь ТОЛЬКО валидным JSON. Строго соблюдай правила:
- Никаких переносов строк внутри значений
- НИКОГДА не используй двойные кавычки " внутри строковых значений — только backtick ` для кода
- Без markdown-обёртки, без объяснений

{{
  "goal": {{
    "id": "goal_{goal_id}",
    "title": "краткое название цели (до 60 символов)",
    "language": "{req.language}",
    "level": "{req.level}",
    "duration_min": {req.duration_min},
    "success_criteria": "критерий успеха одним предложением"
  }},
  "block": {{
    "id": "block_{block_id}",
    "outcomes": ["студент умеет ...", "студент понимает ...", "студент применяет ..."],
    "prerequisites": ["базовый синтаксис Python"],
    "steps": [
      {{"step_id": "s1", "type": "micro_lesson", "title": "Теория: <конкретная тема урока — что объясняем>", "content_md": "Объяснение ключевых концепций: <краткое описание содержания теоретического блока>.", "task_id": null}},
      {{"step_id": "s2", "type": "exercise",     "title": "Разбор примера: <что конкретно разбираем>", "content_md": "Разберём пример: <краткое описание разбираемого примера>.", "task_id": null}},
      {{"step_id": "s3", "type": "task",         "title": "Практическое задание: <что нужно реализовать>", "content_md": "Задание: <краткое описание того, что студент пишет самостоятельно>.", "task_id": null}},
      {{"step_id": "s4", "type": "reflection",   "title": "Рефлексия", "content_md": "Самооценка: что удалось, что было трудным и почему, что сделаешь иначе в следующий раз.", "task_id": null}}
    ]
  }}
}}

ВАЖНО: все шаги (title и content_md) должны быть конкретно про тему студента: "{req.goal_text}".
Не используй абстрактные примеры — только то, что относится к цели студента."""

    try:
        if not req.api_key:
            raise HTTPException(status_code=400, detail="API key is required to generate a learning block.")
        iface = _make_interface(req.api_key, req.api_provider)
        result = await asyncio.to_thread(iface._make_api_request, "/chat/completions", {
            "model":       _model_for_provider(req.api_provider),
            "messages":    [{"role": "user", "content": prompt}],
            "temperature": 0.6,
            "max_tokens":  1500,
        })
        if result and "choices" in result:
            content = result["choices"][0]["message"]["content"].strip()
            # Strip markdown code fences if present
            content = re.sub(r'^```[a-z]*\n?', '', content)
            content = re.sub(r'\n?```$', '', content).strip()
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                clean = _sanitize_llm_json(json_match.group())
                try:
                    data = json.loads(clean)
                except json.JSONDecodeError as je:
                    snippet = clean[max(0, je.pos - 80):je.pos + 40]
                    print(f"⚠️ JSON parse error at char {je.pos}: {je.msg}")
                    print(f"⚠️ Context: {repr(snippet)}")
                    print("⚠️ Trying json_repair fallback...")
                    repaired = repair_json(clean, return_objects=True)
                    if isinstance(repaired, dict) and "goal" in repaired and "block" in repaired:
                        data = repaired
                        print("✅ json_repair succeeded")
                    else:
                        raise
                # Force task_id=None on all steps — LLM sometimes fills this in incorrectly
                for step in (data.get("block") or {}).get("steps") or []:
                    step["task_id"] = None
                # Ensure last step is always reflection — LLM sometimes omits it
                _steps = (data.get("block") or {}).get("steps") or []
                if _steps and _steps[-1].get("type") != "reflection":
                    _steps.append({
                        "step_id": "s_refl",
                        "type": "reflection",
                        "title": "Рефлексия",
                        "content_md": "Самооценка: что удалось, что было трудным и почему, что сделаешь иначе в следующий раз.",
                        "task_id": None
                    })
                # Ensure a 'task' step exists — LLM sometimes omits it
                step_types = [s.get("type") for s in _steps]
                if "task" not in step_types:
                    goal_title = (data.get("goal") or {}).get("title", req.goal_text[:40])
                    task_step = {
                        "step_id": "s3",
                        "type": "task",
                        "title": f"Практическое задание: {goal_title[:40]}",
                        "content_md": f"Напиши код, демонстрирующий понимание темы: {goal_title[:60]}.",
                        "task_id": None,
                    }
                    refl_idx = next((i for i, s in enumerate(_steps) if s.get("type") == "reflection"), len(_steps))
                    _steps.insert(refl_idx, task_step)
                if _validate_learning_block(req, data):
                    data.setdefault("goal", {})["requested_topic"] = req.goal_text
                    data["goal"]["level"] = req.level
                    data["goal"]["duration_min"] = req.duration_min
                    return data
                print("⚠️ Goal build rejected: generated block does not match the selected topic or required structure")
    except Exception as e:
        print(f"⚠️ Goal build LLM error: {e}")

    # Fallback
    return _build_template_block(req)


# ─── POST /api/session/create ─────────────────────────────────
@app.post("/api/session/create")
async def create_goal_session(req: SessionCreateRequest):
    """
    Create a new learning session tied to a goal/block.
    Reuses the existing VirtualTutor session infrastructure.
    """
    if len(session_manager.tutors) >= MAX_ACTIVE_SESSIONS:
        raise HTTPException(status_code=503, detail="Сервис занят: достигнут лимит одновременных сессий. Попробуйте позже.")

    session_id = f"sess_{uuid.uuid4().hex}"
    session_token = secrets.token_urlsafe(32)

    # Register VirtualTutor session (same path as POST /api/sessions)
    tutor = VirtualTutor(session_id, moral_profile_engine=session_manager.moral_profile_engine)
    if req.api_key:
        tutor.interface = _make_interface(req.api_key, req.api_provider)
        _propagate_interface(tutor, tutor.interface)
    session_manager.tutors[session_id] = tutor

    # Store goal context (api_key не хранится — Interface живёт в tutor.interface)
    goal_sessions[session_id] = {
        "goal_id":       req.goal_id,
        "block_id":      req.block_id,
        "current_step":  0,
        "task_id":       None,
        "block":         req.block,
        "goal":          req.goal,
        "self_efficacy": req.self_efficacy,  # Zimmermann SRL Forethought
        "api_provider":  req.api_provider,
        "token_hash":    _token_hash(session_token),
        "chat_history":  [],
        "editor_code":   {},   # {task_id: code} — per-task storage
        "theory_cache":  {},
        "last_test_result": None,
        "test_results":   {},   # {task_id: autograde result}
        "passed_tasks":   {},   # {task_id: bool}; progress cannot leak between tasks
        "reflection_result": None,
    }
    tutor.set_goal_context(goal_sessions[session_id])
    _persist_session(session_id)

    # Запускаем изолированный контейнер для сессии
    if session_manager.docker_client:
        try:
            await session_manager.start_container(session_id)
        except Exception as exc:
            goal_sessions.pop(session_id, None)
            session_manager.tutors.pop(session_id, None)
            session_store.delete(session_id)
            raise HTTPException(status_code=503, detail=str(exc))

    return {
        "session_id":   session_id,
        "ws": {
            "tutor":    f"/ws/tutor/{session_id}",
        },
        "session_token": session_token,
    }


# ─── GET /api/session/{session_id} ───────────────────────────
@app.get("/api/session/{session_id}")
async def get_goal_session(session_id: str, x_session_token: Optional[str] = Header(default=None)):
    """Return resumable session state; access is restricted by an opaque session token."""
    data = _require_session_token(session_id, x_session_token)

    tutor = session_manager.get_tutor(session_id)
    metrics = {}
    if tutor:
        metrics = tutor.metrics

    editor_code_raw = data.get("editor_code", {})
    # Backward compatibility: old sessions stored a single string
    if isinstance(editor_code_raw, str):
        editor_code_raw = {}
    # Return code for the current task, falling back to empty string
    current_task_id = data.get("task_id")
    current_editor_code = editor_code_raw.get(current_task_id, "") if current_task_id else ""

    return {
        "session_id":   session_id,
        "current_step": data.get("current_step", 0),
        "task_id":      current_task_id,
        "goal":         data.get("goal"),
        "block":        data.get("block"),
        "metrics":      metrics,
        "chat_history": data.get("chat_history", []),
        "editor_code":  current_editor_code,
        "editor_code_per_task": editor_code_raw,  # full dict for frontend to manage
        "theory_cache": data.get("theory_cache", {}),
        "last_test_result": data.get("last_test_result"),
        "test_results":     data.get("test_results", {}),
        "last_run_output":  data.get("last_run_output"),
        "reflection_result": data.get("reflection_result"),
    }

@app.post("/api/session/{session_id}/resume")
async def resume_goal_session(session_id: str, req: SessionResumeRequest):
    data = _require_session_token(session_id, req.session_token)
    tutor = session_manager.get_tutor(session_id) or _restore_session(session_id, req.api_key, req.api_provider)
    if tutor and req.api_key:
        tutor.interface = _make_interface(req.api_key, req.api_provider)
        _propagate_interface(tutor, tutor.interface)
    if session_manager.docker_client and session_id not in session_manager.session_containers:
        try:
            await session_manager.start_container(session_id)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc))
    session_manager.session_last_active[session_id] = time.time()
    return {"status": "resumed", "session_id": session_id}

@app.patch("/api/session/{session_id}/state")
async def patch_session_state(session_id: str, req: SessionStatePatch, x_session_token: Optional[str] = Header(default=None)):
    data = _require_session_token(session_id, x_session_token)
    for key, value in req.dict(exclude_none=True).items():
        if key == "editor_code":
            if isinstance(value, dict):
                # Merge: preserve existing per-task codes
                existing = data.get("editor_code", {})
                if isinstance(existing, str):
                    existing = {}
                existing.update(value)
                data[key] = existing
            elif isinstance(value, str):
                # Frontend sends plain string — convert to dict with current task_id
                existing = data.get("editor_code", {})
                if isinstance(existing, str):
                    existing = {}
                current_task_id = data.get("task_id")
                if current_task_id:
                    existing[current_task_id] = value
                data[key] = existing
            else:
                data[key] = value
        else:
            data[key] = value
    _persist_session(session_id)
    return {"status": "saved"}


# ─── POST /api/session/update_step ────────────────────────────
@app.post("/api/session/{session_id}/step")
async def update_session_step(session_id: str, data: dict, x_session_token: Optional[str] = Header(default=None)):
    """Update current step index in a goal session."""
    session = _require_session_token(session_id, x_session_token)
    if session_id in goal_sessions:
        steps = session.get('block', {}).get('steps', [])
        step_idx = int(data.get("step", 0))
        if step_idx < 0 or step_idx >= len(steps):
            raise HTTPException(status_code=400, detail="Invalid step index")
        current_idx = int(session.get("current_step", 0))
        passed_tasks = session.setdefault("passed_tasks", {})
        test_results = session.get("test_results") or {}

        def _task_passed_for_progress(task_id: Optional[str]) -> bool:
            if not task_id:
                return False
            if passed_tasks.get(task_id, False):
                return True
            saved_result = test_results.get(task_id) or {}
            if _autograde_passed_enough(saved_result.get("summary") or {}):
                passed_tasks[task_id] = True
                return True
            return False

        if step_idx > current_idx:
            for prev_idx, prev_step in enumerate(steps[:step_idx]):
                if prev_step.get("type") not in {"exercise", "task"}:
                    continue
                prev_task_id = prev_step.get("task_id")
                if not _task_passed_for_progress(prev_task_id):
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            f"Для перехода к шагу {step_idx + 1} необходимо пройти автопроверку "
                            f"предыдущего задания на шаге {prev_idx + 1}"
                        ),
                    )
        goal_sessions[session_id]["current_step"] = step_idx
        # Определяем тип шага и активное задание; прохождение одной задачи
        # не должно автоматически открывать следующую.
        step_type = steps[step_idx]['type']
        selected_task_id = steps[step_idx].get("task_id") if step_type in {"exercise", "task"} else None
        if selected_task_id:
            goal_sessions[session_id]["task_id"] = selected_task_id
            goal_sessions[session_id]["task_passed"] = _task_passed_for_progress(selected_task_id)
        goal_sessions[session_id]['current_step_type'] = step_type
        goal_sessions[session_id]['srl_tracking'] = {
            'step_started_at': time.time(),
            'attempts_this_step': 0,
            'hints_this_step': 0,
        }
        # Обновляем контекст на тьюторе, чтобы он знал текущий шаг
        tutor = session_manager.get_tutor(session_id)
        if tutor and hasattr(tutor, 'set_goal_context'):
            tutor.set_goal_context(goal_sessions[session_id])
        # Синхронизируем SRL-фазу с типом шага (UI и логи совпадают)
        _srl_phase_map = {
            'micro_lesson': 'planning',
            'exercise':     'performance',
            'task':         'performance',
            'reflection':   'reflection',
        }
        if tutor and hasattr(tutor, 'srl_analyzer'):
            tutor.srl_analyzer.phase_override = _srl_phase_map.get(step_type, 'performance')
        _persist_session(session_id)
    return {"status": "ok"}


# ─── POST /api/session/{session_id}/reflect ───────────────────
@app.post("/api/session/{session_id}/reflect")
async def session_reflect(session_id: str, req: ReflectionRequest, x_session_token: Optional[str] = Header(default=None)):
    """
    SRL Self-Reflection phase (Zimmermann, 2000).
    Accepts student's self-evaluation and returns LLM-generated session debrief.
    """
    session_data = _require_session_token(session_id, x_session_token)
    goal = session_data.get("goal") or {}
    goal_title = goal.get("title") or "программирование на Python"

    # Gather actual metrics from VirtualTutor
    tutor = session_manager.get_tutor(session_id)
    actual_metrics: dict = {}
    if tutor:
        m = tutor.metrics or {}
        block = session_data.get("block") or {}
        total_steps = len((block.get("steps") or []))
        current_step = session_data.get("current_step", 0)
        steps_done = min(current_step + 1, total_steps)
        hint_count = m.get("hint_requests", 0) or m.get("hint_rate", 0)
        # Accuracy: fraction of successful tests
        total_tests = m.get("total_tests", 0)
        passed_tests = m.get("tests_passed", 0)
        accuracy_str = f"{passed_tests}/{total_tests}" if total_tests > 0 else "—"
        actual_metrics = {
            "accuracy": accuracy_str,
            "hints": hint_count,
            "steps_done": f"{steps_done}/{total_steps}" if total_steps else "—",
        }

    # Build LLM prompt for personalised debrief
    prompt = f"""Ты — педагог-наставник. Студент завершил учебную сессию по теме «{goal_title}».

Фактические результаты сессии:
- Пройдено шагов: {actual_metrics.get('steps_done', '—')}
- Точность выполнения задач: {actual_metrics.get('accuracy', '—')}
- Запросов подсказок: {actual_metrics.get('hints', '—')}

Самоотчёт студента (фаза самооценки по Зиммерманну):
- Самооценка (1-5): {req.self_rating}
- Что получилось: «{req.what_worked}»
- Трудности: «{req.difficulty}»
- Следующий раз сделает: «{req.next_time}»

Напиши краткий итог (3-5 предложений) на русском языке:
1. Отметь главное достижение студента на этой сессии.
2. Оцени соответствие самооценки ({req.self_rating}/5) реальным результатам.
3. Дай одну конкретную рекомендацию для следующей сессии на основе трудностей.
Пиши тепло и поддерживающе. Не используй заголовки, только текст."""

    summary = ""
    try:
        iface = _get_session_interface(session_id)
        result = await asyncio.to_thread(iface._make_api_request, "/chat/completions", {
            "model":       _model_for_provider(session_data.get("api_provider", "deepseek")),
            "messages":    [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens":  400,
        })
        if result and "choices" in result:
            summary = result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"⚠️ Reflection LLM error: {e}")
        summary = (
            f"Сессия завершена. Ты прошёл тему «{goal_title}» и оценил себя на {req.self_rating}/5. "
            f"Твои ответы сохранены. Продолжай практиковаться — это лучший способ закрепить знания!"
        )

    session_data["reflection_result"] = {"summary": summary, "actual_metrics": actual_metrics}
    _persist_session(session_id)
    return {"summary": summary, "actual_metrics": actual_metrics}


# ─── GET /api/task/{task_id} ──────────────────────────────────
@app.get("/api/task/{task_id}")
async def get_task(task_id: str, session_id: str, x_session_token: Optional[str] = Header(default=None)):
    """
    Return task statement and starter code from the tasks/ folder.
    Reads files from tutor-data/tasks/{task_id}/ on the host.
    """
    _validate_task_id(task_id)
    data = _require_session_token(session_id, x_session_token)
    if task_id != data.get("task_id"):
        raise HTTPException(status_code=403, detail="Task is not part of this session")
    # Resolve path relative to project root (mounted as tutor-data/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tasks_root = os.path.join(project_root, "tutor-data", "tasks")
    task_dir = os.path.abspath(os.path.join(tasks_root, task_id))
    if not task_dir.startswith(os.path.abspath(tasks_root) + os.sep):
        raise HTTPException(status_code=400, detail="Invalid task_id")

    statement_path    = os.path.join(task_dir, "statement.md")
    starter_code_path = os.path.join(task_dir, "starter_code.py")

    if not os.path.isdir(task_dir):
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")

    statement = ""
    if os.path.exists(statement_path):
        with open(statement_path, "r", encoding="utf-8") as f:
            statement = f.read()

    starter_code = "# Напиши своё решение здесь\n"
    if os.path.exists(starter_code_path):
        with open(starter_code_path, "r", encoding="utf-8") as f:
            starter_code = f.read()

    return {
        "task_id":      task_id,
        "statement":    statement,
        "starter_code": starter_code,
    }


# ─── POST /api/run ────────────────────────────────────────────
@app.post("/api/run")
async def run_code(req: RunRequest, x_session_token: Optional[str] = Header(default=None)):
    """
    Save user code to container and run it.
    Returns stdout/stderr as plain text.
    """
    _validate_task_id(req.task_id)
    session = _require_session_token(req.session_id, x_session_token)
    if req.task_id != session.get("task_id"):
        raise HTTPException(status_code=403, detail="Task is not part of this session")
    handler = _get_handler(req.session_id)
    session_manager.session_last_active[req.session_id] = time.time()

    # Save and execute code only through the controlled sandbox backend.
    try:
        save_result = await _save_code_to_container(handler, req.task_id, req.code)
        if "Error" in (save_result or ''):
            return JSONResponse(status_code=503, content={"output": f"⚠️ Не удалось сохранить файл: {save_result}", "status": "error"})

        # Timeout protects against infinite loops in student code.
        try:
            output = await asyncio.wait_for(
                handler.execute(f"cd /app/tasks/{req.task_id} && timeout -k 1s 10s python3 starter_code.py 2>&1"),
                timeout=10.0
            )
        except asyncio.TimeoutError:
            return {"output": "⏱ Превышено время выполнения (10 с). Проверь наличие бесконечных циклов.", "status": "timeout"}
    except docker.errors.APIError as exc:
        print(f"⚠️ Sandbox Docker API error in /api/run: {exc}")
        return JSONResponse(status_code=503, content={
            "status": "error",
            "error": "Среда выполнения кода временно недоступна. Перезапустите сервисы после обновления docker-compose.yml.",
        })
    except Exception as exc:
        print(f"⚠️ Sandbox error in /api/run: {type(exc).__name__}: {exc}")
        return JSONResponse(status_code=500, content={"status": "error", "error": "Не удалось выполнить код в изолированной среде."})
    # Save code per-task so switching between tasks preserves each one's code.
    editor_code = session.setdefault("editor_code", {})
    if isinstance(editor_code, str):
        editor_code = {}
        session["editor_code"] = editor_code
    editor_code[req.task_id] = req.code
    session["last_run_output"] = output or "(нет вывода)"
    _persist_session(req.session_id)
    return {"output": output or "(нет вывода)", "status": "ok"}


# ─── POST /api/theory/expand ─────────────────────────────────
@app.post("/api/theory/expand")
async def expand_theory(req: TheoryExpandRequest, x_session_token: Optional[str] = Header(default=None)):
    """Generate an additional theory section on demand."""
    session = _require_session_token(req.session_id, x_session_token)
    goal = session.get("goal") or {}
    goal_title = goal.get("title", req.goal_title)
    prompt = (
        f"Ты — методист по обучению Python. Студент изучает тему: \"{req.base_topic}\" (цель: {goal_title}).\n"
        f"Он хочет узнать подробнее: \"{req.user_request}\"\n\n"
        f"Напиши дополнительный раздел теории на русском языке в формате markdown (100-250 слов).\n"
        f"Включи примеры кода в блоках ```python если уместно.\n"
        f"Ответь ТОЛЬКО валидным JSON:\n"
        f"{{\"expansion_md\": \"текст раздела в markdown\"}}"
    )
    try:
        result = await asyncio.to_thread(
            _get_session_interface(req.session_id)._make_api_request,
            "/chat/completions", {
                "model":       _model_for_provider(session.get("api_provider", "deepseek")),
                "messages":    [{"role": "user", "content": prompt}],
                "temperature": 0.5,
                "max_tokens":  800,
            }
        )
        content = result["choices"][0]["message"]["content"].strip()
        content = re.sub(r'^```[a-z]*\n?', '', content)
        content = re.sub(r'\n?```$', '', content).strip()
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            raw_json = json_match.group()
            try:
                expansion_md = json.loads(raw_json).get("expansion_md", "")
            except json.JSONDecodeError:
                repaired = repair_json(raw_json, return_objects=True)
                expansion_md = repaired.get("expansion_md", content) if isinstance(repaired, dict) else content
        else:
            expansion_md = content
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"statement": expansion_md}


# ─── POST /api/autograde ─────────────────────────────────────
@app.post("/api/autograde")
async def autograde(req: AutogradeRequest, x_session_token: Optional[str] = Header(default=None)):
    """
    Run pytest for the given task and return structured results.
    If files are provided, saves them to the container first.
    """
    _validate_task_id(req.task_id)
    session = _require_session_token(req.session_id, x_session_token)
    if req.task_id != session.get("task_id"):
        raise HTTPException(status_code=403, detail="Task is not part of this session")
    handler = _get_handler(req.session_id)
    session_manager.session_last_active[req.session_id] = time.time()

    try:
        # Save provided files. Only the current student's solution may be overwritten.
        if req.files:
            for filename, content in req.files.items():
                if filename in ("starter_code.py", "main.py"):
                    await _save_code_to_container(handler, req.task_id, content)

        # Ensure task files exist in the sandbox container (for generated tasks).
        check = await handler.execute(f"test -d /app/tasks/{req.task_id}/tests && echo OK || echo MISSING")
        if "MISSING" in (check or ""):
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            task_dir = os.path.join(project_root, "tutor-data", "tasks", req.task_id)
            tests_file = os.path.join(task_dir, "tests", "test_main.py")
            starter_file = os.path.join(task_dir, "starter_code.py")
            if os.path.exists(tests_file):
                with open(tests_file, "r", encoding="utf-8") as f:
                    await _save_file_to_container(handler, f"/app/tasks/{req.task_id}/tests/test_main.py", f.read())
            if os.path.exists(starter_file):
                with open(starter_file, "r", encoding="utf-8") as f:
                    await _save_file_to_container(handler, f"/app/tasks/{req.task_id}/starter_code.py", f.read())
            conftest_host = os.path.join(task_dir, "conftest.py")
            conftest_content = "import sys, os\nsys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\n"
            if os.path.exists(conftest_host):
                with open(conftest_host, "r", encoding="utf-8") as f:
                    conftest_content = f.read()
            await _save_file_to_container(handler, f"/app/tasks/{req.task_id}/conftest.py", conftest_content)

        print(f"🧪 autograde: task_id={req.task_id}")
        cmd = (
            f"cd /app/tasks/{req.task_id} && "
            f"timeout -k 1s 20s python3 -m pytest tests/ -v --tb=long --no-header 2>&1"
        )
        try:
            output = await asyncio.wait_for(handler.execute(cmd), timeout=20.0)
        except asyncio.TimeoutError:
            output = "TIMEOUT: pytest не завершился за 20 с"
    except docker.errors.APIError as exc:
        print(f"⚠️ Sandbox Docker API error in /api/autograde: {exc}")
        return JSONResponse(status_code=503, content={
            "status": "error",
            "error": "Среда автопроверки временно недоступна. Перезапустите сервисы после обновления docker-compose.yml.",
            "tests": [],
            "summary": {"passed": 0, "failed": 0, "total": 0},
        })
    except Exception as exc:
        print(f"⚠️ Sandbox error in /api/autograde: {type(exc).__name__}: {exc}")
        return JSONResponse(status_code=500, content={
            "status": "error",
            "error": "Не удалось запустить автопроверку в изолированной среде.",
            "tests": [],
            "summary": {"passed": 0, "failed": 0, "total": 0},
        })
    print(f"🧪 pytest output:\n{output}")

    result = _parse_pytest_output(output or "")

    # Log test result in tutor metrics if session exists
    tutor = session_manager.get_tutor(req.session_id)
    if tutor and result["summary"]["total"] > 0:
        success = _autograde_passed_enough(result.get("summary") or {})
        tutor.update_metrics("test", success)

    # SRL/eBICA: создать TaskInteraction из результатов теста и добавить тьютору
    try:
        _session    = goal_sessions.get(req.session_id, {})
        _tracking   = _session.get('srl_tracking', {})
        _step_type  = _session.get('current_step_type', 'task')
        _phase_map  = {
            'micro_lesson': 'planning',
            'exercise':     'performance',
            'task':         'performance',
            'reflection':   'reflection',
        }
        _now        = time.time()
        _summary    = result["summary"]
        _passed     = _summary.get("passed", 0)
        _total      = _summary.get("total", 0)
        _passed_enough = _autograde_passed_enough(_summary)
        _interaction = TaskInteraction(
            session_id    = req.session_id,
            task_id       = req.task_id,
            timestamp     = _now,
            phase         = _phase_map.get(_step_type, 'performance'),
            success       = _passed_enough,
            used_hint     = _tracking.get('hints_this_step', 0) > 0,
            attempts      = _tracking.get('attempts_this_step', 0) + 1,
            response_time = _now - _tracking.get('step_started_at', _now),
            error         = (_total == 0),
        )
        # Добавить в список тьютора для SRLAnalyzer
        _tutor = session_manager.get_or_create_tutor(req.session_id, use_moral=True)
        _tutor.task_interactions.append(_interaction)
        # Залогировать в DataRepository
        _repo = get_data_repository()
        _repo.log_event('srl', {
            'session_id':  req.session_id,
            'interaction': interaction_to_dict(_interaction),
        })
        # Обновить счётчик попыток
        _tracking['attempts_this_step'] = _tracking.get('attempts_this_step', 0) + 1
    except Exception as _srl_err:
        print(f"⚠️ SRL tracking error (non-critical): {_srl_err}")

    # Обновляем моральный профиль только фактически полученным результатом автопроверки.
    try:
        session_manager.moral_profile_engine.update_profile(
            req.session_id,
            {
                "type": "task_result",
                "success": _autograde_passed_enough(result.get("summary") or {}),
                "used_hint": bool((session.get("srl_tracking") or {}).get("hints_this_step", 0)),
            },
        )
    except Exception as exc:
        print(f"⚠️ profile update after autograde failed: {exc}")

    # Сохраняем только фактически полученный результат автопроверки.
    session["last_test_result"] = result
    session.setdefault("test_results", {})[req.task_id] = result
    passed = _autograde_passed_enough(result.get("summary") or {})
    session.setdefault("passed_tasks", {})[req.task_id] = passed
    session["task_passed"] = passed
    _persist_session(req.session_id)
    return result


# ─── Helper: write file to container (generic) ──────────────
async def _save_file_to_container(handler: TerminalHandler, filepath: str, content: str) -> str:
    """Safely writes generated materials to the restricted /app/tasks namespace."""
    return await handler.save_file(filepath, content)


# ─── POST /api/task/generate ─────────────────────────────────
@app.post("/api/task/generate")
async def generate_task(req: TaskGenerateRequest, x_session_token: Optional[str] = Header(default=None)):
    """
    Generate content based on step type:
    - micro_lesson  → LLM generates theory explanation (no files/tests)
    - reflection    → return content_md as-is (no LLM call)
    - task/exercise → generate code task + pytest tests (existing logic)
    """
    session = _require_session_token(req.session_id, x_session_token)
    goal = session.get("goal") or {}
    authoritative_goal = goal.get("requested_topic") or goal.get("title") or req.goal_title
    authoritative_level = goal.get("level") or req.level
    req.goal_title = authoritative_goal
    req.level = authoritative_level

    # ── Theory: micro_lesson ─────────────────────────────────────────────────
    if req.step_type == "micro_lesson":
        theory_specs = {
            "beginner": {
                "label": "новичок",
                "words": "650-900 слов",
                "min_words": 520,
                "focus": "объясняй без жаргона, вводи термины постепенно, показывай простые примеры и типичные ошибки новичка",
            },
            "intermediate": {
                "label": "средний уровень",
                "words": "750-1050 слов",
                "min_words": 620,
                "focus": "связывай тему с функциями, коллекциями, декомпозицией, edge-cases и читаемостью кода",
            },
            "advanced": {
                "label": "продвинутый уровень",
                "words": "850-1200 слов",
                "min_words": 700,
                "focus": "добавь глубину: ограничения подходов, trade-offs, сложность, архитектурные нюансы и нетривиальные примеры",
            },
        }
        theory_spec = theory_specs.get(req.level, theory_specs["beginner"])

        def _theory_word_count(text: str) -> int:
            return len(re.findall(r"[A-Za-zА-Яа-яЁё0-9_]+", text or ""))

        theory_prompt = f"""Ты — методист по обучению Python. Создай большой, содержательный теоретический блок для студента.

Цель студента: {req.goal_title}
Тема шага: {req.step_title}
Описание: {req.step_content_md}
Уровень: {theory_spec["label"]}

Требования к теории:
- Объём: {theory_spec["words"]}. Не делай краткую справку.
- Теория должна раскрывать именно цель студента: "{req.goal_title}", а не общую тему Python.
- Адаптация к уровню: {theory_spec["focus"]}.
- Структура markdown: `## Зачем это нужно`, `## Ключевые идеи`, `## Как это работает`, `## Пример кода`, `## Частые ошибки`, `## Мини-проверка понимания`.
- Включи минимум 2 блока кода ```python: один короткий базовый пример и один пример ближе к практической задаче.
- После примеров объясни построчно или по смысловым блокам, что происходит.
- В конце дай 3 вопроса для самопроверки.

Ответь ТОЛЬКО валидным JSON (без markdown, без объяснений):
{{
  "theory_md": "Большое объяснение на русском языке в формате markdown. Используй \\n для переносов строк внутри этой JSON-строки."
}}"""
        try:
            iface = _get_session_interface(req.session_id)
            model_name = _model_for_provider(session.get("api_provider", "deepseek"))
            theory_md = ""
            prompt = theory_prompt
            for attempt in range(1, 3):
                result = await asyncio.to_thread(iface._make_api_request, "/chat/completions", {
                    "model":       model_name,
                    "messages":    [{"role": "user", "content": prompt}],
                    "temperature": 0.45 + 0.1 * (attempt - 1),
                    "max_tokens":  2600,
                })
                if result and "choices" in result:
                    content = result["choices"][0]["message"]["content"].strip()
                    content = re.sub(r'^```[a-z]*\n?', '', content)
                    content = re.sub(r'\n?```$', '', content).strip()
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        try:
                            theory_md = json.loads(_sanitize_llm_json(json_match.group())).get("theory_md", "")
                        except json.JSONDecodeError:
                            repaired = repair_json(json_match.group(), return_objects=True)
                            theory_md = repaired.get("theory_md", "") if isinstance(repaired, dict) else ""
                else:
                    print(f"⚠️ Theory LLM returned None/empty for: {req.step_title}. result={result}")
                if _theory_word_count(theory_md) >= theory_spec["min_words"]:
                    break
                prompt = (
                    theory_prompt
                    + f"\n\nПредыдущая версия была слишком короткой ({_theory_word_count(theory_md)} слов). "
                    + f"Раскрой тему подробнее, минимум {theory_spec['min_words']} слов, сохрани уровень «{theory_spec['label']}»."
                )
            if _theory_word_count(theory_md) < 180:
                theory_md = (
                    f"## Теория: {req.goal_title}\n\n"
                    f"{req.step_content_md or req.step_title}\n\n"
                    "Не удалось получить полный теоретический блок от языковой модели. "
                    "Нажми «Расширить теорию» и уточни, какую часть разобрать подробнее."
                )
        except Exception as e:
            print(f"⚠️ Theory generate error ({type(e).__name__}): {e}")
            theory_md = req.step_content_md or req.step_title

        return {
            "task_id":      None,
            "statement":    theory_md,
            "starter_code": "# Попробуй концепцию в редакторе\n",
            "step_type":    req.step_type,
            "status":       "ok",
        }

    # ── Reflection: return content_md directly ───────────────────────────────
    if req.step_type == "reflection":
        reflection_text = req.step_content_md or "Подведи итоги: что ты понял на этом занятии? Где были трудности?"
        return {
            "task_id":      None,
            "statement":    reflection_text,
            "starter_code": "# Запиши свои мысли или поэкспериментируй с кодом\n",
            "step_type":    req.step_type,
            "status":       "ok",
        }

    # ── Code task: task / exercise ───────────────────────────────────────────
    task_id = f"gen_{uuid.uuid4().hex[:8]}"

    step_context = req.step_content_md if req.step_content_md.strip() else f"Практика по теме: {req.goal_title}"

    # Detailed level specs: allowed, forbidden, required constructs, expected volume
    level_specs = {
        "beginner": {
            "label":           "Новичок",
            "allowed":         "переменные, условия if/else, циклы for/while, простые функции с 1-2 параметрами",
            "forbidden":       "классы, list comprehensions, lambda, декораторы, try/except, рекурсия",
            "required":        "",
            "exercise_volume": "4–6 строк в теле функции",
            "task_volume":     "8–14 строк в теле функции",
        },
        "intermediate": {
            "label":           "Средний",
            "allowed":         "функции, списки, словари, строковые методы, простые классы, базовые исключения",
            "forbidden":       "декораторы, метаклассы, async/await, сложные паттерны ООП",
            "required":        "хотя бы одну из: операции со списками/словарями, строковые методы или простой класс — только if/else недостаточно",
            "exercise_volume": "8–14 строк в теле функции",
            "task_volume":     "18–30 строк в теле функции",
        },
        "advanced": {
            "label":           "Продвинутый",
            "allowed":         "любые конструкции Python: классы, декораторы, генераторы, comprehensions, рекурсия, ООП",
            "forbidden":       "нет ограничений",
            "required":        "ОБЯЗАТЕЛЬНО использовать ООП или функциональное программирование: класс с методами, наследование, декоратор, генератор, comprehension или рекурсия — простая функция с if/else недопустима",
            "exercise_volume": "18–30 строк в теле функции",
            "task_volume":     "35–60 строк в теле функции",
        },
    }
    spec = level_specs.get(req.level, level_specs["beginner"])

    # Duration-based volume scaling
    session_goal = goal_sessions.get(req.session_id, {}).get("goal", {})
    duration_min = int(session_goal.get("duration_min", 45))
    if duration_min <= 20:
        dur_factor, dur_note = 0.6, f"короткая сессия ({duration_min} мин) — компактное задание"
    elif duration_min >= 80:
        dur_factor, dur_note = 1.5, f"длинная сессия ({duration_min} мин) — полноценное задание"
    else:
        dur_factor, dur_note = 1.0, f"стандартная сессия ({duration_min} мин)"

    raw_volume = spec["exercise_volume"] if req.step_type == "exercise" else spec["task_volume"]
    _vm = re.match(r"(\d+)[–\-](\d+)(.*)", raw_volume)
    if _vm and dur_factor != 1.0:
        lo = max(1, round(int(_vm.group(1)) * dur_factor))
        hi = max(lo + 2, round(int(_vm.group(2)) * dur_factor))
        scaled_volume = f"{lo}–{hi}{_vm.group(3)}"
    else:
        scaled_volume = raw_volume

    required_line = f"  🔴 ОБЯЗАТЕЛЬНО использовать: {spec['required']}\n" if spec["required"] else ""

    # Step-type + level complexity note
    if req.step_type == "exercise":
        if req.level == "advanced":
            complexity_note = (
                "ТИП: Упражнение для уровня Продвинутый (первое кодовое задание).\n"
                "Задача ОБЯЗАНА использовать ООП или функциональное программирование.\n"
                "Проще чем практическое задание, но требует класс, comprehension или рекурсию."
            )
        elif req.level == "intermediate":
            complexity_note = (
                "ТИП: Упражнение для уровня Средний (первое кодовое задание).\n"
                "Одна функция или простой класс, использующий списки/словари или строковые методы."
            )
        else:
            complexity_note = (
                "ТИП: Упражнение для уровня Новичок (первое кодовое задание — ПРОЩЕ).\n"
                "Одна простая функция с 1-2 параметрами, прямолинейная логика без вложенных условий."
            )
    else:
        if req.level == "advanced":
            complexity_note = (
                "ТИП: Практическое задание для уровня Продвинутый (второе задание — СЛОЖНЕЕ).\n"
                "Требует нескольких классов или нетривиальной логики. ООП или FP обязательны, много edge-cases."
            )
        elif req.level == "intermediate":
            complexity_note = (
                "ТИП: Практическое задание для уровня Средний (второе задание — СЛОЖНЕЕ).\n"
                "Несколько функций или класс с методами, обработка исключений, больше edge-cases."
            )
        else:
            complexity_note = (
                "ТИП: Практическое задание для уровня Новичок (второе задание — СЛОЖНЕЕ).\n"
                "Несколько условий или цикл с аккумулятором, нетривиальная логика для новичка."
            )

    def _parse_llm_json(raw: str) -> dict:
        raw = re.sub(r'^```[a-z]*\n?', '', raw.strip())
        raw = re.sub(r'\n?```$', '', raw).strip()
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if not m:
            raise ValueError("LLM response is not valid JSON")
        blob = m.group()
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            repaired = repair_json(blob, return_objects=True)
            if isinstance(repaired, dict):
                return repaired
            raise ValueError(f"json_repair failed: {repr(repaired)[:80]}")

    iface = _get_session_interface(req.session_id)

    model_name = _model_for_provider(session.get("api_provider", "deepseek"))

    # Base prompt (Call 1: statement + skeleton starter). Escalation text is
    # appended on retries when a task turns out too easy for the level.
    base_task_prompt = f"""Ты — методист по обучению программированию на Python.
Сгенерируй ОДНО учебное задание СТРОГО по теме: "{req.goal_title}".

{complexity_note}

Тема: {req.goal_title}
Шаг: {req.step_title}
Описание: {step_context}
Уровень: {spec['label']}
  ✅ Разрешено: {spec['allowed']}
  ❌ Запрещено в решении: {spec['forbidden']}
{required_line}  📏 Ожидаемый объём решения: {scaled_volume} ({dur_note})

Ответь ТОЛЬКО валидным JSON (без markdown, без объяснений):
{{
  "statement_md": "Условие задания на русском в markdown (2-5 предложений).",
  "starter_code": "# Стартовый код\\ndef func_name(...):\\n    \\\"\\\"\\\"Docstring.\\\"\\\"\\\"\\n    # TODO: реализуй функцию\\n    pass\\n\\n\\nif __name__ == '__main__':\\n    print(func_name(...))"
}}

ВАЖНО для starter_code:
- Тело функции: ТОЛЬКО docstring + `# TODO: реализуй функцию` + `pass`
- ❌ ЗАПРЕЩЕНО писать готовое решение — студент пишет код сам
- Все строки экранированы (\\n для переносов)"""

    def _build_test_prompt(stmt: str) -> str:
        return f"""Ты — методист по обучению программированию на Python.
Напиши pytest-тесты для следующего задания.

Задание: {stmt}
Тема: {req.goal_title}
Уровень: {spec['label']}

Студент реализует функцию в файле starter_code.py.

Ответь ТОЛЬКО валидным JSON (без markdown, без объяснений):
{{
  "test_code": "import pytest\\nimport sys, os\\nsys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))\\nfrom starter_code import func_name\\n\\ndef test_basic():\\n    assert func_name(...) == ...\\n\\ndef test_edge():\\n    assert func_name(...) == ..."
}}

ВАЖНО:
- 4-6 функций def test_...
- Импорт: from starter_code import func_name (имя функции из задания)
- Тесты покрывают базовые случаи И edge-cases
- Все строки экранированы (\\n для переносов)"""

    # ── Difficulty-gated generation with auto-regeneration (problem §3/§5) ────
    # Each attempt: LLM produces statement+skeleton (call 1) and tests (call 2);
    # we then check structural relevance AND programmatic difficulty. A task
    # that is too simple for the level is regenerated with an escalated prompt
    # instead of being shown. A weak last attempt must not silently bypass the
    # requested level, otherwise an advanced user still receives a trivial task.
    MAX_GEN_ATTEMPTS = 3
    accepted = None        # (statement_md, starter_code, test_code, notes)
    escalation = ""
    last_error = None
    last_problems: list[str] = []

    for attempt in range(1, MAX_GEN_ATTEMPTS + 1):
        prompt = base_task_prompt + (f"\n\n⚠️ {escalation}" if escalation else "")
        try:
            r1 = await asyncio.wait_for(
                asyncio.to_thread(iface._make_api_request, "/chat/completions", {
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.5 + 0.1 * (attempt - 1),
                    "max_tokens": 1000,
                }),
                timeout=30.0,
            )
            if not (r1 and "choices" in r1):
                raise ValueError("No LLM response (call 1)")
            task_data = _parse_llm_json(r1["choices"][0]["message"]["content"])
            statement_md = task_data.get("statement_md", "")
            starter_code = _strip_function_bodies(task_data.get("starter_code", ""))

            r2 = await asyncio.wait_for(
                asyncio.to_thread(iface._make_api_request, "/chat/completions", {
                    "model": model_name,
                    "messages": [{"role": "user", "content": _build_test_prompt(statement_md)}],
                    "temperature": 0.3,
                    "max_tokens": 800,
                }),
                timeout=30.0,
            )
            if not (r2 and "choices" in r2):
                raise ValueError("No LLM response (call 2)")
            test_data = _parse_llm_json(r2["choices"][0]["message"]["content"])
            test_code = test_data.get("test_code", "")
        except Exception as e:
            last_error = f"LLM error ({type(e).__name__}): {e}"
            print(f"⚠️ generate_task [{req.step_type}] attempt {attempt}/{MAX_GEN_ATTEMPTS} FAILED: {last_error}")
            continue

        task_valid, task_validation_notes = _validate_task_relevance(
            authoritative_goal, statement_md, starter_code, authoritative_level)
        meets_difficulty, difficulty_reasons = _assess_task_difficulty(
            authoritative_level, statement_md, starter_code, test_code, duration_min)
        problems = list(task_validation_notes)
        if not test_code.strip():
            problems.append("отсутствуют тесты автопроверки")
        if not meets_difficulty:
            problems += difficulty_reasons

        if task_valid and test_code.strip() and meets_difficulty:
            accepted = (statement_md, starter_code, test_code, task_validation_notes)
            if task_validation_notes:
                print(f"⚠️ Task validation note for {task_id}: {'; '.join(task_validation_notes)}")
            break

        last_problems = problems
        print(f"♻️ generate_task attempt {attempt}/{MAX_GEN_ATTEMPTS} too weak "
              f"({'; '.join(problems) or 'n/a'}); regenerating")
        escalation = (
            f"Предыдущая попытка не подходит для уровня «{spec['label']}»: "
            f"{'; '.join(problems)}. Сделай задание заметно содержательнее и сложнее, "
            f"добавь больше тестовых случаев и обработку граничных ситуаций, "
            f"строго сохраняя тему «{req.goal_title}»."
        )

    if accepted is None:
        if last_error and not last_problems:
            raise HTTPException(status_code=502, detail="Не удалось получить корректный ответ от языковой модели.")
        details = "; ".join(last_problems) or "задание не прошло проверку сложности"
        raise HTTPException(
            status_code=422,
            detail=(
                f"Не удалось сформировать содержательное задание уровня «{spec['label']}» "
                f"по выбранной теме после {MAX_GEN_ATTEMPTS} попыток: {details}."
            ),
        )

    statement_md, starter_code, test_code, task_validation_notes = accepted
    if not statement_md.strip() or not starter_code.strip() or not test_code.strip():
        raise HTTPException(
            status_code=422,
            detail="Сгенерированное задание отклонено: пустое условие, стартовый код или тесты.")

    # Write files to host filesystem
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    task_dir = os.path.join(project_root, "tutor-data", "tasks", task_id)
    tests_dir = os.path.join(task_dir, "tests")
    os.makedirs(tests_dir, exist_ok=True)

    conftest_content = "import sys, os\nsys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\n"
    with open(os.path.join(task_dir, "statement.md"), "w", encoding="utf-8") as f:
        f.write(statement_md)
    with open(os.path.join(task_dir, "starter_code.py"), "w", encoding="utf-8") as f:
        f.write(starter_code)
    with open(os.path.join(tests_dir, "test_main.py"), "w", encoding="utf-8") as f:
        f.write(test_code)
    with open(os.path.join(task_dir, "conftest.py"), "w", encoding="utf-8") as f:
        f.write(conftest_content)

    # Write files to Docker container
    try:
        handler = TerminalHandler(session_manager.docker_client, session_manager.get_container_name(req.session_id))
        container_task_dir = f"/app/tasks/{task_id}"
        await _save_file_to_container(handler, f"{container_task_dir}/statement.md", statement_md)
        await _save_file_to_container(handler, f"{container_task_dir}/starter_code.py", starter_code)
        await _save_file_to_container(handler, f"{container_task_dir}/tests/test_main.py", test_code)
        await _save_file_to_container(handler, f"{container_task_dir}/conftest.py", conftest_content)
    except Exception as e:
        print(f"⚠️ Could not write task to container (will copy on autograde): {e}")

    # Update goal_sessions and persist the task identifier inside the relevant
    # StudyBlock step so a page reload can restore every generated task.
    if req.session_id in goal_sessions:
        current_session = goal_sessions[req.session_id]
        current_session["task_id"] = task_id
        current_session.setdefault("passed_tasks", {})[task_id] = False
        current_session["task_passed"] = False
        steps = (current_session.get("block") or {}).get("steps") or []
        if 0 <= req.step_index < len(steps):
            steps[req.step_index]["task_id"] = task_id
        _persist_session(req.session_id)

    return {
        "task_id":      task_id,
        "statement":    statement_md,
        "starter_code": starter_code,
        "status":       "ok",
    }


# ─────────────────────────────────────────────────────────────
# END NEW ENDPOINTS
# ─────────────────────────────────────────────────────────────


# ═════════════════════════════════════════════════════════════
# MISSING ENDPOINTS: WebSocket Tutor, Metrics, Affective
# ═════════════════════════════════════════════════════════════


# ─── WS /ws/tutor/{session_id} ──────────────────────────────
@app.websocket("/ws/tutor/{session_id}")
async def tutor_websocket(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for tutor chat.
    Accepts connection, authenticates via token query param, then
    streams LLM-generated tutor responses back to the client.
    Protocol:
      - Client sends a text message (user's chat message).
      - Server responds with:
          STREAM_START
          STREAM_CHUNK:<text>
          STREAM_DONE
      - Ping/pong keepalive handled automatically.
    """
    if not _ws_authorized(session_id, websocket):
        await websocket.close(code=4001, reason="Unauthorized")
        return

    await websocket.accept()
    session_manager.active_connections[session_id] = websocket
    session_manager.session_last_active[session_id] = time.time()

    try:
        while True:
            message = await websocket.receive_text()

            # Ping keepalive — just acknowledge
            if message.strip().lower() == "ping":
                continue

            # Update activity timestamp
            session_manager.session_last_active[session_id] = time.time()

            # Generate tutor response
            tutor = session_manager.get_or_create_tutor(session_id, use_moral=True)
            # Sync goal context
            if session_id in goal_sessions:
                tutor.set_goal_context(goal_sessions[session_id])

            # Send streaming start
            await websocket.send_text("STREAM_START")
            # Always initialize reply: if generation fails before assignment,
            # chat history persistence below must not close the WebSocket with
            # UnboundLocalError.
            reply = ""

            try:
                # Generate full reply via VirtualTutor
                reply = await asyncio.to_thread(tutor.generate_answer, message)

                # Simulate streaming by sending the full reply in chunks
                if reply:
                    # Send in ~50-char chunks for realistic streaming feel
                    chunk_size = 50
                    for i in range(0, len(reply), chunk_size):
                        chunk = reply[i:i + chunk_size]
                        await websocket.send_text(f"STREAM_CHUNK:{chunk}")
                        await asyncio.sleep(0.02)  # small delay for streaming effect
                else:
                    await websocket.send_text("STREAM_CHUNK:Извини, я не смог сформулировать ответ. Попробуй ещё раз.")

            except Exception as e:
                print(f"⚠️ Tutor WS generate error: {e}")
                await websocket.send_text("STREAM_CHUNK:Произошла ошибка при генерации ответа. Попробуйте ещё раз.")

            await websocket.send_text("STREAM_DONE")

            # Save chat history
            if session_id in goal_sessions:
                history = goal_sessions[session_id].setdefault("chat_history", [])
                history.append({"role": "user", "content": message})
                if reply:
                    history.append({"role": "assistant", "content": reply})
                if len(history) > 100:
                    history[:] = history[-100:]
                _persist_session(session_id)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"⚠️ Tutor WS error: {e}")
    finally:
        session_manager.active_connections.pop(session_id, None)


# ─── GET /api/metrics/{session_id} ───────────────────────────
@app.get("/api/metrics/{session_id}")
async def get_session_metrics(session_id: str, x_session_token: Optional[str] = Header(default=None)):
    """
    Return tutor metrics for a session.
    Used by the frontend analytics dashboard.
    """
    data = _require_session_token(session_id, x_session_token)
    tutor = session_manager.get_tutor(session_id)
    if not tutor:
        return {
            "metrics": {},
            "adaptive_settings": {},
            "progress": {},
        }

    metrics = dict(getattr(tutor, "metrics", {}) or {})
    adaptive = dict(getattr(tutor, "adaptive_settings", {}) or {})

    # Compute progress summary
    progress = {
        "current_stage": getattr(tutor, "cur_moral_id", 0) + 1,
        "total_stages": len(getattr(tutor, "ms_list", []) or []),
        "schemes_completed": sum(getattr(tutor, "schemes", []) or []),
        "session_duration": time.time() - metrics.get("session_start_time", time.time()),
    }

    return {
        "metrics": metrics,
        "adaptive_settings": adaptive,
        "progress": progress,
    }


# ─── GET /api/affective/{session_id} ─────────────────────────
@app.get("/api/affective/{session_id}")
async def get_session_affective(session_id: str, x_session_token: Optional[str] = Header(default=None)):
    """
    Return the last known affective state (FS, ESI) for a session.
    """
    data = _require_session_token(session_id, x_session_token)
    tutor = session_manager.get_tutor(session_id)
    if not tutor:
        return {"state": {}, "last_update": None}

    affective = dict(getattr(tutor, "last_affective_state", {}) or {})

    return {
        "state": affective,
        "last_update": time.time(),
    }
