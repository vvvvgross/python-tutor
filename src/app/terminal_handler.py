"""Restricted container runner used only by controlled backend endpoints.

No public interactive terminal is exposed. File operations are restricted to the
current generated task directory under /app/tasks.
"""
from __future__ import annotations

import asyncio
import base64
import posixpath
import re
from typing import Optional

import docker

_SAFE_TASK_PATH = re.compile(r"^/app/tasks/[A-Za-z0-9_-]+(?:/[A-Za-z0-9_.-]+)*$")


class TerminalHandler:
    def __init__(self, docker_client, container_name: str):
        self.docker_client = docker_client
        self.container_name = container_name
        self.container = None

    async def get_container(self):
        if not self.container or self.container.status != "running":
            try:
                self.container = await asyncio.to_thread(self.docker_client.containers.get, self.container_name)
                if self.container.status != "running":
                    await asyncio.to_thread(self.container.start)
            except docker.errors.NotFound:
                return None
            except Exception as exc:
                print(f"Container error ({self.container_name}): {exc}")
                return None
        return self.container

    async def execute(self, command: str) -> str:
        """Execute a backend-authored command; never expose this method to the browser."""
        container = await self.get_container()
        if not container:
            return "Docker container not available"
        result = await asyncio.to_thread(container.exec_run, ["sh", "-c", command], demux=True, workdir="/app")
        stdout = result.output[0].decode(errors="replace") if result.output[0] else ""
        stderr = result.output[1].decode(errors="replace") if result.output[1] else ""
        if result.exit_code != 0:
            return stdout if stdout else f"Command failed ({result.exit_code}): {stderr or 'No error output'}"
        return stdout

    @staticmethod
    def _validate_path(filename: str) -> str:
        normalized = posixpath.normpath(filename)
        if not _SAFE_TASK_PATH.fullmatch(normalized):
            raise ValueError("File access outside the current task directory is forbidden")
        return normalized

    async def get_file_content(self, filename: str) -> str:
        filename = self._validate_path(filename)
        container = await self.get_container()
        if not container:
            return "Docker container not available"
        result = await asyncio.to_thread(container.exec_run, ["cat", filename], demux=True, workdir="/app")
        if result.exit_code != 0:
            return f"File not found or cannot be read: {filename}"
        return result.output[0].decode(errors="replace") if result.output[0] else ""

    async def save_file(self, filename: str, content: str) -> str:
        filename = self._validate_path(filename)
        container = await self.get_container()
        if not container:
            return "Docker container not available"
        directory, basename = posixpath.split(filename)
        # Create the task directory first using exec_run (works reliably on
        # tmpfs in read_only containers).
        mk = await asyncio.to_thread(
            container.exec_run, ["mkdir", "-p", directory], workdir="/app", demux=True
        )
        if mk.exit_code != 0:
            err = (mk.output[1] or mk.output[0] or b"").decode(errors="replace").strip()
            raise RuntimeError(
                f"Cannot create directory {directory!r} in sandbox "
                f"(exit {mk.exit_code}): {err or 'unknown error'}"
            )
        # Use exec_run + base64 to write file content instead of put_archive.
        # put_archive fails on read_only containers with tmpfs (Docker API
        # returns 404 "Could not find the file") because the directory created
        # via exec_run lives in the tmpfs namespace that put_archive cannot
        # resolve at the overlay level. Writing via exec_run uses the same
        # execution context and works reliably.
        encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
        write_cmd = f"echo '{encoded}' | base64 -d > {filename}"
        wr = await asyncio.to_thread(
            container.exec_run, ["sh", "-c", write_cmd], workdir="/app", demux=True
        )
        if wr.exit_code != 0:
            err = (wr.output[1] or wr.output[0] or b"").decode(errors="replace").strip()
            return f"Error saving file: {err or 'write failed'}"
        return "File saved successfully"
