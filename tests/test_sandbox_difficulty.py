"""Regression tests for the two fixes in this round:

1. Sandbox file write (`TerminalHandler.save_file`) must use `exec_run` +
   base64 instead of `put_archive` — the root cause of the /api/run & /api/autograde
   "Could not find the file /app/tasks/<task_id> in container" errors.

2. Programmatic difficulty gating (`_assess_task_difficulty`) must reject
   tasks that are clearly too easy for the requested level so the generator
   can regenerate them, while staying lenient enough not to block valid tasks.
"""
import asyncio
import os
import sys
from collections import namedtuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from app.terminal_handler import TerminalHandler
from test_server import (
    _assess_task_difficulty,
    _count_test_cases,
    _DIFFICULTY_FLOORS,
)

_ExecResult = namedtuple("ExecResult", ["exit_code", "output"])


class _FakeContainer:
    """Minimal stand-in mimicking the docker-py container API used by save_file."""

    def __init__(self, mkdir_exit_code=0, mkdir_stderr=b"", write_exit_code=0):
        self.status = "running"
        self.mkdir_exit_code = mkdir_exit_code
        self.mkdir_stderr = mkdir_stderr
        self.write_exit_code = write_exit_code
        self.exec_calls = []

    def exec_run(self, cmd, workdir=None, demux=False):
        self.exec_calls.append(cmd)
        if cmd[:1] == ["mkdir"]:
            return _ExecResult(self.mkdir_exit_code, (b"", self.mkdir_stderr))
        if cmd[:1] == ["sh"]:
            return _ExecResult(self.write_exit_code, (b"", b""))
        return _ExecResult(0, (b"", b""))


def _make_handler(container):
    handler = TerminalHandler(docker_client=None, container_name="tutor_sess_x")
    handler.container = container  # bypass docker_client.containers.get
    return handler


# ─── Sandbox file write ──────────────────────────────────────────────────────

def test_save_file_writes_when_directory_is_writable():
    container = _FakeContainer(mkdir_exit_code=0)
    handler = _make_handler(container)
    result = asyncio.run(handler.save_file("/app/tasks/gen_abc/starter_code.py", "print(1)\n"))
    assert result == "File saved successfully"
    # Must have called mkdir first, then base64 write via sh -c.
    assert len(container.exec_calls) == 2
    assert container.exec_calls[0][:1] == ["mkdir"]
    assert container.exec_calls[1][:1] == ["sh"]


def test_save_file_raises_when_mkdir_fails():
    # Simulates the old bug: workspace not writable → mkdir fails → previously
    # put_archive silently 404'd. Now the failure is raised with a clear cause.
    container = _FakeContainer(mkdir_exit_code=1, mkdir_stderr=b"Permission denied")
    handler = _make_handler(container)
    try:
        asyncio.run(handler.save_file("/app/tasks/gen_abc/starter_code.py", "print(1)\n"))
    except RuntimeError as exc:
        assert "Permission denied" in str(exc)
        assert "/app/tasks/gen_abc" in str(exc)
    else:
        raise AssertionError("save_file should raise when mkdir fails")
    # And it must NOT attempt to write the file (no second exec).
    assert len(container.exec_calls) == 1  # only mkdir
    assert container.exec_calls[0][:1] == ["mkdir"]


# ─── Difficulty gating ───────────────────────────────────────────────────────

def test_count_test_cases():
    code = (
        "def test_basic():\n    assert f(1) == 1\n"
        "def test_edge():\n    assert f(0) == 0\n"
        "def helper():\n    pass\n"
        "def test_more():\n    assert f(2) == 2\n"
    )
    assert _count_test_cases(code) == 3


def _tests(n):
    return "".join(f"def test_{i}():\n    assert f({i}) == {i}\n" for i in range(n))


def test_beginner_reasonable_task_accepted():
    ok, reasons = _assess_task_difficulty(
        "beginner",
        "Напишите функцию, которая принимает список чисел и возвращает их сумму с помощью цикла.",
        "def total(nums):\n    pass\n",
        _tests(3),
        duration_min=30,
    )
    assert ok, reasons


def test_advanced_trivial_task_rejected():
    # A one-function add(a, b) with few tests is too weak for advanced.
    ok, reasons = _assess_task_difficulty(
        "advanced",
        "Сложите два числа.",
        "def add(a, b):\n    pass\n",
        _tests(2),
        duration_min=45,
    )
    assert not ok
    assert reasons


def test_advanced_class_task_accepted():
    ok, reasons = _assess_task_difficulty(
        "advanced",
        "Реализуйте класс Stack с методами push, pop и проверкой переполнения; "
        "обработайте граничные случаи пустого стека и используйте рекурсию для обхода.",
        "class Stack:\n    def push(self, x):\n        pass\n    def pop(self):\n        pass\n",
        _tests(5),
        duration_min=60,
    )
    assert ok, reasons


def test_intermediate_pure_arithmetic_rejected():
    ok, reasons = _assess_task_difficulty(
        "intermediate",
        "Умножьте число на два.",
        "def double(x):\n    pass\n",
        _tests(4),
        duration_min=45,
    )
    assert not ok
    assert reasons


def test_difficulty_floors_increase_with_level():
    assert _DIFFICULTY_FLOORS["beginner"]["min_tests"] <= _DIFFICULTY_FLOORS["intermediate"]["min_tests"]
    assert _DIFFICULTY_FLOORS["intermediate"]["min_words"] <= _DIFFICULTY_FLOORS["advanced"]["min_words"]
