from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

REPO = Path(__file__).resolve().parents[1]


def module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


class ModelReplContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = module(
            "model_repl_runner_test", "tests/integration/repl/run_model_repl_qemu.py"
        )
        cls.topology = module("event_topology_test", "scripts/event_topology.py")

    def test_all_printable_ascii_has_qmp_mapping_and_controls_do_not(self):
        for value in range(0x20, 0x7f):
            qcode, shifted = self.runner.qcode_for_ascii(chr(value))
            self.assertTrue(qcode)
            self.assertIs(type(shifted), bool)
        for value in (0, 9, 10, 13, 31, 127, 128):
            with self.assertRaises(ValueError):
                self.runner.qcode_for_ascii(chr(value))

    def test_ctrl_c_injection_holds_control_around_c(self):
        class Session:
            def __init__(self):
                self.arguments = None

            def command(self, name, _deadline, arguments):
                self.assert_name = name
                self.arguments = arguments
                return {"return": {}}

        session = Session()
        self.runner.inject(session, time.monotonic() + 1.0, "c", control=True)
        self.assertEqual(session.assert_name, "input-send-event")
        self.assertEqual(
            session.arguments["events"],
            [
                {"type": "key", "data": {"down": True, "key": {"type": "qcode", "data": "ctrl"}}},
                {"type": "key", "data": {"down": True, "key": {"type": "qcode", "data": "c"}}},
                {"type": "key", "data": {"down": False, "key": {"type": "qcode", "data": "c"}}},
                {"type": "key", "data": {"down": False, "key": {"type": "qcode", "data": "ctrl"}}},
            ],
        )

    def test_serial_redraw_ranges_are_removed_without_touching_other_bytes(self):
        self.assertEqual(
            self.runner.without_ranges(b"012345", [(1, 3), (4, 5)]),
            b"035",
        )
        for ranges in ([(2, 1)], [(2, 4), (3, 5)], [(0, 7)]):
            with self.assertRaises(ValueError):
                self.runner.without_ranges(b"012345", ranges)

    def test_reset_oracle_requires_two_identical_fresh_turns(self):
        oracle = json.loads(
            (REPO / "fixtures/reference/repl/eos-reset.json").read_text(
                encoding="ascii"
            )
        )
        first, second = oracle["turns"]
        self.assertEqual((first["prompt_index"], second["prompt_index"]), (7, 9))
        self.assertEqual(first["prompt_tokens"], second["prompt_tokens"])
        self.assertEqual(first["generated"], second["generated"])
        self.assertEqual((first["reason"], second["reason"]), ("EOS", "EOS"))

    def test_record_schema_accepts_reordering_and_diagnostics(self):
        valid = b"PROMPTBOOT_EVENT v=1 event=INPUT_ACCEPTED prompt_index=1 bytes=5 accepted_tsc=0000000000000000\r\n"
        reordered = b"PROMPTBOOT_EVENT v=1 event=INPUT_ACCEPTED bytes=5 prompt_index=1 accepted_tsc=0000000000000000\r\n"
        additive = valid.removesuffix(b"\r\n") + b" diagnostic=useful\r\n"
        self.runner.validate_record_schemas([valid, reordered, additive])

    def test_live_wait_and_selection_match_semantic_fields(self):
        record = (
            b"PROMPTBOOT_EVENT v=1 diagnostic=useful history_tokens=0 "
            b"event=PROMPT_READY reserve=1024 prompt_index=3 "
            b"history_turns=0 input_limit=127 sampling_draws=0\r\n"
        )

        class Process:
            returncode = None

            @staticmethod
            def poll():
                return None

        with tempfile.TemporaryDirectory() as temporary:
            serial = Path(temporary) / "com1.log"
            serial.write_bytes(record)
            observed = self.runner.wait_record(
                serial,
                "PROMPT_READY",
                time.monotonic() + 1.0,
                Process(),
                self.topology.TOGGLE_V1,
                prompt_index=3,
            )
            self.assertLessEqual(observed, time.monotonic())

        self.assertEqual(
            self.runner.matching_records(
                [record],
                "PROMPT_READY",
                prompt_index=3,
                history_turns=0,
            ),
            [record],
        )
        self.assertEqual(
            self.runner.matching_records(
                [record], "PROMPT_READY", prompt_index=2
            ),
            [],
        )

    def test_record_schema_rejects_missing_duplicate_malformed_and_unknown(self):
        invalid = (
            b"PROMPTBOOT_EVENT v=1 prompt_index=1 bytes=5 accepted_tsc=0\r\n",
            b"PROMPTBOOT_EVENT v=1 event=INPUT_ACCEPTED prompt_index=1 bytes=5\r\n",
            b"PROMPTBOOT_EVENT v=1 event=INPUT_ACCEPTED prompt_index=1 prompt_index=2 bytes=5 accepted_tsc=0\r\n",
            b"PROMPTBOOT_EVENT v=1 event=INPUT_ACCEPTED prompt_index bytes=5 accepted_tsc=0\r\n",
            b"PROMPTBOOT_EVENT v=1 event=INPUT_ACCEPTED prompt_index=1 bytes= accepted_tsc=0\r\n",
            b"PROMPTBOOT_EVENT v=1 event=INPUT_ACCEPTED bad-key=1 prompt_index=1 bytes=5 accepted_tsc=0\r\n",
            b"PROMPTBOOT_EVENT v=1 event=SOMETHING_NEW detail=1\r\n",
        )
        for record in invalid:
            with self.subTest(record=record), self.assertRaises(ValueError):
                self.runner.validate_record_schemas([record])

    def test_toggle_topology_accepts_pairs_and_rejects_bad_boundaries(self):
        record_a = b"PROMPTBOOT_EVENT v=1 event=A\r\n"
        record_b = b"PROMPTBOOT_EVENT v=1 event=B\r\n"
        record_c = b"PROMPTBOOT_EVENT v=1 event=C\r\n"
        valid = (
            record_a
            + b"events: on\r\n"
            + record_b
            + record_b
            + b"events: off\r\n"
            + record_c
        )
        parsed = self.topology.parse_records(
            valid, self.topology.TOGGLE_V1, self.topology.STRICT_FINAL
        )
        self.assertEqual(parsed.logical_records, (record_a, record_b, record_c))
        self.assertEqual(parsed.physical_records, (record_a, record_b, record_b, record_c))
        self.assertIsNone(parsed.pending_record)
        for payload in (
            record_a + record_a,
            b"events: on\r\n" + record_b + record_c,
            b"events: on\r\n" + record_b + b"x" + record_b,
            b"events: off\r\n" + record_a,
            b"events: on\r\n" + record_b,
        ):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                self.topology.parse_records(
                    payload, self.topology.TOGGLE_V1, self.topology.STRICT_FINAL
                )
        polling = self.topology.parse_records(
            b"events: on\r\n" + record_b,
            self.topology.TOGGLE_V1,
            self.topology.POLLING_PREFIX,
        )
        self.assertEqual(polling.logical_records, ())
        self.assertEqual(polling.physical_records, (record_b,))
        self.assertEqual(polling.pending_record, record_b)
        with self.assertRaises(ValueError):
            self.topology.parse_records(
                record_a, "unknown-v1", self.topology.STRICT_FINAL
            )

    def test_evidence_directory_clears_only_known_run_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary) / "nested" / "run"
            evidence.mkdir(parents=True)
            stale = evidence / "outcome.json"
            unrelated = evidence / "notes.txt"
            stale.write_text("stale", encoding="ascii")
            unrelated.write_text("keep", encoding="ascii")
            self.assertEqual(
                self.runner.prepare_evidence_directory(evidence),
                evidence.resolve(),
            )
            self.assertFalse(stale.exists())
            self.assertEqual(unrelated.read_text(encoding="ascii"), "keep")
            self.runner.prepare_evidence_directory(evidence)

    def test_runner_binds_external_manifest_to_embedded_build_jsn(self):
        manifest_bytes = b'{"event_topology":"toggle-v1","mode":"model_repl"}\n'
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            esp = root / "model.img"
            manifest = root / "BUILD.JSN"
            esp.write_bytes(b"fixture")
            manifest.write_bytes(manifest_bytes)

            def inspector(path, external_manifest, distribution_root):
                self.assertEqual(
                    (path, external_manifest, distribution_root),
                    (esp, manifest, None),
                )
                return {
                    "kind": "positive",
                    "build_jsn_sha256": hashlib.sha256(manifest_bytes).hexdigest()
                }

            loaded, report = self.runner.load_bound_inputs(
                esp, manifest, inspector=inspector
            )
            self.assertEqual(loaded["event_topology"], "toggle-v1")
            self.assertEqual(
                report["build_jsn_sha256"], hashlib.sha256(manifest_bytes).hexdigest()
            )
            manifest.write_bytes(manifest_bytes + b" ")
            with self.assertRaisesRegex(ValueError, "external manifest does not match"):
                self.runner.load_bound_inputs(esp, manifest, inspector=inspector)

    def test_release_image_inspection_uses_distribution_files_not_release_claims(self):
        manifest_bytes = b'{"event_topology":"toggle-v1","mode":"model_repl"}\n'
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            esp = root / "promptboot.img"
            manifest = root / "BUILD.JSN"
            esp.write_bytes(b"fixture")
            manifest.write_bytes(manifest_bytes)
            (root / "release.json").write_text(
                '{"distribution":{"untrusted":"claim"}}\n', encoding="ascii"
            )
            calls = []

            def inspector(path, external_manifest, distribution_root):
                calls.append((path, external_manifest, distribution_root))
                return {
                    "kind": "positive",
                    "build_jsn_sha256": hashlib.sha256(manifest_bytes).hexdigest()
                }

            self.runner.load_bound_inputs(esp, manifest, inspector=inspector)
            self.assertEqual(len(calls), 1)
            path, external_manifest, distribution_root = calls[0]
            self.assertEqual((path, external_manifest), (esp, manifest))
            self.assertEqual(distribution_root, root)




if __name__ == "__main__":
    unittest.main()
