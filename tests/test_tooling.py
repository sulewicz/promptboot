from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import host_check


class HostDiscoveryTests(unittest.TestCase):
    def test_explicit_firmware_environment_wins(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code, variables = root / "code.fd", root / "vars.fd"
            code.write_bytes(b"code")
            variables.write_bytes(b"vars")
            self.assertEqual(
                host_check.discover_firmware(
                    environ={"OVMF_CODE": str(code), "OVMF_VARS": str(variables)},
                    candidates=(),
                ),
                (code.resolve(), variables.resolve()),
            )

    def test_fedora_and_debian_candidate_layouts_are_discovered(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = (root / "missing-code", root / "missing-vars")
            for layout in ("fedora", "debian"):
                with self.subTest(layout=layout):
                    code, variables = (
                        root / layout / "code.fd",
                        root / layout / "vars.fd",
                    )
                    code.parent.mkdir()
                    code.write_bytes(b"code")
                    variables.write_bytes(b"vars")
                    self.assertEqual(
                        host_check.discover_firmware(
                            environ={},
                            candidates=(missing, (code, variables)),
                        ),
                        (code.resolve(), variables.resolve()),
                    )


if __name__ == "__main__":
    unittest.main()
