from __future__ import annotations

import errno
import importlib.util
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


qmp = load("focused_qmp_session", REPO / "scripts/qmp_session.py")
repl = load(
    "focused_repl_runner", REPO / "tests/integration/repl/run_model_repl_qemu.py"
)


class FakeSocket:
    def __init__(self, connect_error=None, stream=None):
        self.timeout = None
        self.timeouts = []
        self.closed = False
        self.connect_error = connect_error
        self.stream = stream

    def settimeout(self, value):
        self.timeout = value
        self.timeouts.append(value)

    def close(self):
        self.closed = True

    def connect(self, _path):
        if self.connect_error is not None:
            raise self.connect_error

    def makefile(self, _mode, buffering=0):
        self.buffering = buffering
        return self.stream


class ScriptedStream:
    def __init__(
        self,
        frames=(),
        *,
        write_results=(),
        flush_error=None,
        close_error=None,
    ):
        self._sock = FakeSocket()
        self.frames = list(frames)
        self.write_results = list(write_results)
        self.flush_error = flush_error
        self.close_error = close_error
        self.writes = []
        self.accepted = bytearray()
        self.flushes = 0
        self.closed = False

    def readline(self):
        if not self.frames:
            raise socket.timeout()
        frame = self.frames.pop(0)
        if isinstance(frame, BaseException):
            raise frame
        return frame

    def write(self, value):
        value = bytes(value)
        self.writes.append(value)
        if self.write_results:
            result = self.write_results.pop(0)
            if isinstance(result, BaseException):
                raise result
            if isinstance(result, int) and result > 0:
                self.accepted.extend(value[:result])
            return result
        self.accepted.extend(value)
        return len(value)

    def flush(self):
        self.flushes += 1
        if self.flush_error is not None:
            raise self.flush_error

    def close(self):
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


def scripted(frames=(), **kwargs):
    stream = ScriptedStream(frames, **kwargs)
    return qmp.QmpSession(None, stream), stream


class QmpSessionTests(unittest.TestCase):
    def test_real_af_unix_readiness_and_retry_sockets_close(self):
        for error in (
            FileNotFoundError(),
            ConnectionRefusedError(),
            socket.timeout(),
        ):
            with self.subTest(error=type(error).__name__):
                clients = []

                def factory(_family, _kind):
                    client = FakeSocket(connect_error=error)
                    clients.append(client)
                    return client

                with mock.patch.object(
                    qmp.time, "monotonic", side_effect=[0.0, 0.0, 1.0, 1.0]
                ):
                    with self.assertRaisesRegex(TimeoutError, "connect timeout"):
                        qmp.QmpSession.connect(
                            Path("unused"),
                            1.0,
                            socket_factory=factory,
                            sleep=lambda _delay: None,
                        )
                self.assertTrue(clients)
                self.assertTrue(all(client.closed for client in clients))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            delayed = root / "delayed.sock"
            observed = []
            failure = []
            ready = threading.Event()

            def serve():
                try:
                    with socket.socket(
                        socket.AF_UNIX, socket.SOCK_STREAM
                    ) as server:
                        server.bind(str(delayed))
                        server.listen(1)
                        ready.set()
                        connection, _ = server.accept()
                        with connection:
                            connection.sendall(
                                b'{"QMP":{"version":{},"capabilities":[]}}\r\n'
                            )
                            observed.append(connection.recv(4096))
                            connection.sendall(b'{"return":{}}\r\n')
                except BaseException as error:
                    failure.append(error)
                    ready.set()

            thread = threading.Thread(target=serve)
            thread.start()
            self.assertTrue(ready.wait(timeout=1.0))
            session = qmp.QmpSession.connect(
                delayed, time.monotonic() + 1.0
            )
            session.close()
            thread.join(timeout=1.0)
            self.assertFalse(thread.is_alive())
            self.assertEqual(failure, [])
            self.assertEqual(
                observed, [b'{"execute":"qmp_capabilities"}\n']
            )

    def test_exact_greeting_capability_bytes_and_malformed_shapes(self):
        session, stream = scripted([
            b'{"QMP":{"version":{},"capabilities":[]}}\n',
            b'{"return":{}}\n',
        ])
        session.handshake(time.monotonic() + 0.1)
        self.assertEqual(
            stream.writes, [b'{"execute":"qmp_capabilities"}\n']
        )
        self.assertTrue(session.trace[1]["capability"])

        greetings = (
            b"{}\n",
            b'{"QMP":[]}\n',
            b'{"QMP":{"version":[],"capabilities":[]}}\n',
            b'{"QMP":{"version":{},"capabilities":{}}}\n',
            b'{"QMP":{"version":{},"capabilities":[]},"event":"BAD"}\n',
        )
        for greeting in greetings:
            with self.subTest(greeting=greeting):
                session, _ = scripted([greeting])
                with self.assertRaisesRegex(ValueError, "greeting"):
                    session.handshake(time.monotonic() + 0.1)

        greeting = b'{"QMP":{"version":{},"capabilities":[]}}\n'
        for response, message in (
            (b"{}\n", "missing return"),
            (b'{"error":{}}\n', "error response"),
            (b'{"return":{"unexpected":1}}\n', "nonempty return"),
        ):
            with self.subTest(response=response):
                session, _ = scripted([greeting, response])
                with self.assertRaisesRegex(ValueError, message):
                    session.handshake(time.monotonic() + 0.1)

    def test_compact_commands_without_and_with_arguments_skip_events(self):
        session, stream = scripted([
            b'{"event":"RESET"}\n',
            b'{"return":{}}\n',
            b'{"return":{"ok":1}}\n',
        ])
        self.assertEqual(
            session.command("stop", time.monotonic() + 0.1),
            {"return": {}},
        )
        self.assertEqual(
            session.command(
                "query",
                time.monotonic() + 0.1,
                {"name": "caf\u00e9"},
            ),
            {"return": {"ok": 1}},
        )
        self.assertEqual(
            stream.writes,
            [
                b'{"execute":"stop"}\n',
                b'{"execute":"query","arguments":{"name":"caf\\u00e9"}}\n',
            ],
        )
        self.assertEqual(stream.flushes, 2)

    def test_command_error_missing_return_and_malformed_combinations(self):
        for frame, message in (
            (b'{"error":{"class":"GenericError"}}\n', "error response"),
            (b"{}\n", "missing return"),
            (b'{"return":{},"error":{}}\n', "response frame is malformed"),
            (b'{"event":"STOP","return":{}}\n', "event frame is malformed"),
            (b'{"event":"STOP","error":{}}\n', "event frame is malformed"),
            (b'{"event":7}\n', "event frame is malformed"),
        ):
            with self.subTest(frame=frame):
                session, _ = scripted([frame])
                with self.assertRaisesRegex(ValueError, message):
                    session.command("test", time.monotonic() + 0.1)

    def test_invalid_utf8_json_scalar_partial_stall_deadline_and_disconnect(self):
        for frame, message in (
            (b"\xff\n", "not valid UTF-8 JSON"),
            (b"{bad}\n", "not valid UTF-8 JSON"),
            (b"[]\n", "not an object"),
            (b'{"return":{}}', "incomplete frame"),
            (b"", "unexpected EOF"),
            (
                ConnectionResetError(errno.ECONNRESET, "reset"),
                "disconnected",
            ),
        ):
            with self.subTest(frame=frame):
                session, _ = scripted([frame])
                with self.assertRaisesRegex(ValueError, message):
                    session.read_frame(time.monotonic() + 0.1)

        session, _ = scripted([socket.timeout()])
        with self.assertRaisesRegex(TimeoutError, "response timeout"):
            session.read_frame(time.monotonic() + 0.01)
        session, _ = scripted([b'{"return":{}}\n'])
        with self.assertRaisesRegex(TimeoutError, "response timeout"):
            session.read_frame(time.monotonic() - 1)

    def test_partial_none_failed_stalled_and_expired_writes(self):
        session, stream = scripted(
            [b'{"return":{}}\n'], write_results=[2]
        )
        session.command("stop", time.monotonic() + 0.1)
        self.assertEqual(
            bytes(stream.accepted), b'{"execute":"stop"}\n'
        )
        self.assertEqual(stream.flushes, 1)

        for result, message in (
            (None, "disconnected during write"),
            (0, "disconnected during write"),
            (OSError(errno.EIO, "write"), "write"),
        ):
            with self.subTest(result=result):
                session, _ = scripted(write_results=[result])
                with self.assertRaisesRegex((ValueError, OSError), message):
                    session.command("stop", time.monotonic() + 0.1)

        session, _ = scripted(write_results=[socket.timeout()])
        with self.assertRaisesRegex(TimeoutError, "write timeout"):
            session.command("stop", time.monotonic() + 0.1)
        session, _ = scripted(flush_error=OSError(errno.EIO, "flush"))
        with self.assertRaisesRegex(OSError, "flush"):
            session.command("stop", time.monotonic() + 0.1)
        session, _ = scripted()
        with self.assertRaisesRegex(TimeoutError, "write timeout"):
            session.command("stop", time.monotonic() - 1)

    def test_partial_writes_and_flush_recheck_absolute_deadline(self):
        session, stream = scripted(write_results=[2, 2])
        with mock.patch.object(
            qmp.time, "monotonic", side_effect=[1.0, 5.0, 11.0]
        ):
            with self.assertRaisesRegex(TimeoutError, "write timeout"):
                session._write(b"abcdef", 10.0, "slow")
        self.assertEqual(stream._sock.timeouts, [9.0, 5.0])
        self.assertEqual(len(stream.writes), 2)
        self.assertEqual(bytes(stream.accepted), b"abcd")
        self.assertEqual(stream.flushes, 0)

        session, stream = scripted(write_results=[2])
        with mock.patch.object(
            qmp.time, "monotonic", side_effect=[1.0, 3.0, 4.0]
        ):
            session._write(b"abcdef", 10.0, "success")
        self.assertEqual(stream._sock.timeouts, [9.0, 7.0, 6.0])
        self.assertEqual(bytes(stream.accepted), b"abcdef")
        self.assertEqual(stream.flushes, 1)

    def test_drain_trailing_non_event_event_eof_and_post_exit_reset(self):
        session, _ = scripted([
            b'{"event":"STOP"}\n',
            b'{"event":"SHUTDOWN"}\n',
            b"",
        ])
        session.drain(time.monotonic() + 0.1)
        session, _ = scripted([
            ConnectionResetError(errno.ECONNRESET, "reset")
        ])
        session.drain(time.monotonic() + 0.1, reset_ok=True)
        session, _ = scripted([b'{"return":{}}\n'])
        with self.assertRaisesRegex(ValueError, "trailing non-event"):
            session.drain(time.monotonic() + 0.1)

    def test_quit_acknowledgement_missing_and_error(self):
        for frame, message in (
            (b"{}\n", "quit missing return"),
            (b'{"error":{"class":"GenericError"}}\n', "quit error response"),
        ):
            session, _ = scripted([frame])
            with self.assertRaisesRegex(ValueError, message):
                session.command("quit", time.monotonic() + 0.1)

    def test_close_is_idempotent_and_closes_client_after_stream_error(self):
        stream = ScriptedStream(close_error=OSError("stream close"))
        client = FakeSocket()
        session = qmp.QmpSession(client, stream)
        with self.assertRaisesRegex(OSError, "stream close"):
            session.close()
        self.assertTrue(stream.closed)
        self.assertTrue(client.closed)
        session.close()

    def test_trace_retains_raw_frames_and_parsed_values(self):
        session, _ = scripted([
            b'{"QMP":{"version":{},"capabilities":[]}}\r\n',
            b'{"return":{}}\r\n',
            b'{"return":{"ok":1}}\r\n',
        ])
        session.handshake(time.monotonic() + 0.1)
        session.command("query", time.monotonic() + 0.1)
        self.assertTrue(
            all(
                str(item["raw"]).endswith("\r")
                for item in session.trace
                if item["direction"] == "receive"
            )
        )
        self.assertEqual(
            session.trace[-1]["value"], {"return": {"ok": 1}}
        )

    def test_runner_shutdown_acknowledges_quit_once_and_rejects_nonzero(self):
        class Session:
            def __init__(self):
                self.commands = []

            def command(self, name, _deadline):
                self.commands.append(name)
                return {"return": {}}

        class Process:
            def __init__(self, returncode=None):
                self.returncode = returncode
                self.waits = 0

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                self.waits += 1
                self.returncode = 0
                return 0

        session = Session()
        process = Process()
        shutdown = {"quit_acknowledged": False}
        repl.terminate_qemu(
            session, process, time.monotonic() + 1.0, shutdown
        )
        repl.terminate_qemu(
            session, process, time.monotonic() + 1.0, shutdown
        )
        self.assertEqual(session.commands, ["quit"])
        self.assertTrue(shutdown["quit_acknowledged"])
        self.assertEqual(process.waits, 1)

        process = Process(returncode=7)
        with self.assertRaisesRegex(RuntimeError, "rc=7"):
            repl.terminate_qemu(
                None,
                process,
                time.monotonic() + 1.0,
                {"quit_acknowledged": False},
            )

    def test_runner_shutdown_terminates_then_kills_after_timeouts(self):
        class Session:
            def __init__(self):
                self.commands = []

            def command(self, name, _deadline):
                self.commands.append(name)
                return {"return": {}}

        class Process:
            def __init__(self):
                self.returncode = None
                self.waits = 0
                self.terminated = False
                self.killed = False

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                self.waits += 1
                if self.waits < 3:
                    raise subprocess.TimeoutExpired("qemu", timeout)
                self.returncode = 0
                return 0

            def terminate(self):
                self.terminated = True

            def kill(self):
                self.killed = True

        session = Session()
        process = Process()
        shutdown = {"quit_acknowledged": False}
        repl.terminate_qemu(
            session, process, time.monotonic() + 1.0, shutdown
        )
        self.assertEqual(session.commands, ["quit"])
        self.assertTrue(process.terminated)
        self.assertTrue(process.killed)
        self.assertEqual(process.waits, 3)



if __name__ == "__main__":
    unittest.main()
