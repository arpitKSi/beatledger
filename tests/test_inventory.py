import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from beatledger.inventory import main


FIXTURE = Path(__file__).parent / "fixtures" / "episode-017.txt"


class InventoryTests(unittest.TestCase):
    def test_cli_writes_coverage_and_source_discrepancies_without_screenplay_text(self):
        source = FIXTURE.read_bytes().replace(b"00:42.00", b"00:16.00")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "episode-017.TXT").write_bytes(source)
            (root / "not-script.json").write_text("{}", encoding="utf-8")
            output = root / "reports" / "inventory.json"
            self.assertEqual(main(["--input", str(root), "--output", str(output)]), 0)
            rendered = output.read_text(encoding="utf-8")

        report = json.loads(rendered)
        self.assertEqual(
            report["summary"],
            {
                "files": 1, "parsed_files": 1, "rejected_files": 0, "files_with_issues": 1,
                "beats": 5, "scenes": 2, "adjacent_overlaps": 3, "backward_starts": 1,
                "warnings": 1, "errors": 0, "parsed_episode_numbers": [17],
            },
        )
        self.assertNotIn("Why are there two keys", rendered)
        self.assertEqual(report["files"][0]["issues"][0]["code"], "beat_outside_scene_clock")

    def test_partial_parse_has_an_error_status_and_keeps_its_report(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "malformed.txt"
            path.write_bytes(FIXTURE.read_bytes().replace(b"00:04.03", b"00:99.03"))
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(main(["--input", str(path)]), 1)

        report = json.loads(stdout.getvalue())
        self.assertEqual(report["summary"]["parsed_files"], 1)
        self.assertEqual(report["summary"]["errors"], 1)
        self.assertEqual(report["files"][0]["beats"], 4)

    def test_refuses_to_overwrite_existing_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "existing.json"
            path.write_text("original", encoding="utf-8")
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                main(["--input", str(FIXTURE), "--output", str(path)])
            self.assertEqual(raised.exception.code, 2)
            self.assertEqual(path.read_text(encoding="utf-8"), "original")
