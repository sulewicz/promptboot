#!/usr/bin/env python3
"""Bounded, dependency-free QMP transport and protocol session."""

from __future__ import annotations

from datetime import datetime, timezone
import errno
import json
from pathlib import Path
import socket
import time
from typing import BinaryIO


class QmpSession:
    def __init__(self, client: socket.socket | None, stream: BinaryIO):
        self.client = client
        self.stream: BinaryIO | None = stream
        self.trace: list[dict[str, object]] = []

    @classmethod
    def connect(
        cls,
        path: Path,
        deadline: float,
        *,
        socket_factory=socket.socket,
        sleep=time.sleep,
    ) -> "QmpSession":
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("QMP connect timeout")
            client = socket_factory(socket.AF_UNIX, socket.SOCK_STREAM)
            stream = None
            try:
                client.settimeout(_remaining(deadline, "QMP connect timeout"))
                client.connect(str(path))
                stream = client.makefile("rwb", buffering=0)
            except (FileNotFoundError, ConnectionRefusedError, socket.timeout):
                if stream is not None:
                    stream.close()
                client.close()
                delay = min(0.02, max(0.0, deadline - time.monotonic()))
                if delay:
                    sleep(delay)
                continue
            except Exception:
                if stream is not None:
                    stream.close()
                client.close()
                raise
            session = cls(client, stream)
            try:
                session.handshake(deadline)
            except Exception:
                session.close()
                raise
            return session

    def _set_timeout(self, deadline: float, message: str) -> None:
        timeout = _remaining(deadline, message)
        if self.client is not None:
            self.client.settimeout(timeout)
            return
        stream = self._stream()
        stream_socket = getattr(stream, "_sock", None)
        if stream_socket is not None:
            stream_socket.settimeout(timeout)

    def _stream(self) -> BinaryIO:
        if self.stream is None:
            raise ValueError("QMP session is closed")
        return self.stream

    def _record(
        self,
        direction: str,
        raw: str,
        value: dict[str, object],
        *,
        capability: bool | None = None,
    ) -> None:
        item: dict[str, object] = {
            "direction": direction,
            "raw": raw,
            "value": value,
            "monotonic": time.monotonic(),
            "monotonic_ns": time.monotonic_ns(),
            "utc": datetime.now(timezone.utc).isoformat(),
        }
        if direction == "send":
            item["capability"] = bool(capability)
        self.trace.append(item)

    def read_frame(
        self,
        deadline: float,
        *,
        eof_ok: bool = False,
        reset_ok: bool = False,
    ) -> dict[str, object] | None:
        self._set_timeout(deadline, "QMP response timeout")
        try:
            line = self._stream().readline()
        except (socket.timeout, TimeoutError) as error:
            raise TimeoutError("QMP response timeout") from error
        except ConnectionResetError as error:
            if reset_ok and error.errno in (None, errno.ECONNRESET):
                return None
            raise ValueError("QMP disconnected") from error
        except OSError as error:
            raise ValueError("QMP disconnected") from error
        if not line:
            if eof_ok:
                return None
            raise ValueError("QMP unexpected EOF")
        if not line.endswith(b"\n"):
            raise ValueError("QMP incomplete frame")
        try:
            raw = line.decode("utf-8").removesuffix("\n")
            value = json.loads(line)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("QMP frame is not valid UTF-8 JSON") from error
        if not isinstance(value, dict):
            raise ValueError("QMP frame is not an object")
        self._record("receive", raw, value)
        return value

    @staticmethod
    def _validate_event(value: dict[str, object]) -> bool:
        if "event" not in value:
            if "return" in value and "error" in value:
                raise ValueError("QMP response frame is malformed")
            return False
        if (
            not isinstance(value["event"], str)
            or "return" in value
            or "error" in value
        ):
            raise ValueError("QMP event frame is malformed")
        return True

    def receive(self, deadline: float) -> dict[str, object]:
        while True:
            value = self.read_frame(deadline)
            assert value is not None
            if not self._validate_event(value):
                return value

    @staticmethod
    def require_return(
        response: dict[str, object], context: str
    ) -> dict[str, object]:
        if "error" in response:
            raise ValueError(f"QMP {context} error response")
        if set(response) != {"return"}:
            raise ValueError(f"QMP {context} missing return")
        return response

    def _write(
        self,
        raw: bytes,
        deadline: float,
        context: str,
    ) -> None:
        timeout_message = f"QMP command {context} write timeout"
        remaining = memoryview(raw)
        try:
            while remaining:
                self._set_timeout(deadline, timeout_message)
                count = self._stream().write(remaining)
                if count is None or count <= 0:
                    raise ValueError("QMP disconnected during write")
                remaining = remaining[count:]
            self._set_timeout(deadline, timeout_message)
            self._stream().flush()
        except (socket.timeout, TimeoutError) as error:
            raise TimeoutError(timeout_message) from error

    def command(
        self,
        name: str,
        deadline: float,
        arguments: dict[str, object] | None = None,
        *,
        capability: bool = False,
    ) -> dict[str, object]:
        request: dict[str, object] = {"execute": name}
        if arguments is not None:
            request["arguments"] = arguments
        raw = (
            json.dumps(
                request, ensure_ascii=True, separators=(",", ":")
            ).encode("ascii")
            + b"\n"
        )
        self._record(
            "send",
            raw.decode("ascii").removesuffix("\n"),
            request,
            capability=capability,
        )
        self._write(raw, deadline, name)
        response = self.receive(deadline)
        return self.require_return(response, name)

    def handshake(self, deadline: float) -> None:
        greeting = self.read_frame(deadline)
        qmp = None if greeting is None else greeting.get("QMP")
        if (
            not isinstance(qmp, dict)
            or not isinstance(qmp.get("version"), dict)
            or not isinstance(qmp.get("capabilities"), list)
            or "event" in greeting
            or "return" in greeting
            or "error" in greeting
        ):
            raise ValueError("QMP greeting missing QMP object")
        response = self.command(
            "qmp_capabilities", deadline, capability=True
        )
        if response != {"return": {}}:
            raise ValueError("QMP qmp_capabilities nonempty return")

    def drain(self, deadline: float, *, reset_ok: bool = False) -> None:
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("QMP EOF timeout")
            try:
                value = self.read_frame(
                    deadline, eof_ok=True, reset_ok=reset_ok
                )
            except TimeoutError as error:
                raise TimeoutError("QMP EOF timeout") from error
            if value is None:
                return
            if "event" not in value:
                raise ValueError("QMP unexpected trailing non-event frame")
            self._validate_event(value)

    def close(self) -> None:
        stream, client = self.stream, self.client
        self.stream = None
        self.client = None
        try:
            if stream is not None:
                stream.close()
        finally:
            if client is not None:
                client.close()


def _remaining(deadline: float, message: str) -> float:
    value = deadline - time.monotonic()
    if value <= 0:
        raise TimeoutError(message)
    return value
