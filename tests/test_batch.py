import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from shutil import copyfile

from run_batch import main


FIXTURES = Path(__file__).parent / "fixtures" / "stitches"


class BatchTests(unittest.TestCase):
    def test_batch_exports_verified_records_and_recap_trace(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "baseline"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--input", str(FIXTURES), "--output", str(output)]), 0)
            ranked = json.loads((output / "ranked_promos.json").read_text(encoding="utf-8"))
            ledger = json.loads((output / "run_ledger.json").read_text(encoding="utf-8"))
            audit = (output / "failures_and_flags.md").read_text(encoding="utf-8")
            with (output / "ranked_promos.csv").open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))

        records = ranked["promos"]
        self.assertEqual(ranked["score_source"], "deterministic_baseline")
        self.assertEqual(ledger["inputs"]["summary"]["parsed_episode_numbers"], [101, 102])
        self.assertEqual(ledger["model_usage"]["requests"], 0)
        self.assertEqual(len(rows), len(records))
        self.assertTrue(all(record["tension_delta"] is None for record in records))

        stitch = next(record for record in records if record["episodes"] == [101, 102])
        part = stitch["segments"][1]
        self.assertEqual((part["start"], part["start_beat_id"]), ("00:05.00", "E102-B003"))
        self.assertEqual(stitch["duration_cs"], sum(p["end_cs"] - p["start_cs"] for p in stitch["segments"]))
        self.assertIn(stitch["candidate_id"], audit)
        row = next(row for row in rows if row["candidate_id"] == stitch["candidate_id"])
        self.assertEqual(row["hook_beat_id"], stitch["hook"]["beat_id"])
        self.assertEqual(row["dialogue_wps"], f'{stitch["dialogue_wps"]:.2f}')
        self.assertEqual(row["tension_delta"], "")

    def test_partial_pack_is_reported_and_existing_run_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            copyfile(FIXTURES / "episode-101.txt", root / "episode-101.txt")
            (root / "mistake.txt").write_text("No episode header", encoding="utf-8")
            output = root / "baseline"
            args = ["--input", str(root), "--output", str(output)]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 1)
            ledger_file = output / "run_ledger.json"
            original = ledger_file.read_bytes()
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                main(args)
            self.assertEqual(raised.exception.code, 2)
            self.assertEqual(ledger_file.read_bytes(), original)

        summary = json.loads(original)["inputs"]["summary"]
        self.assertEqual((summary["files"], summary["parsed_files"], summary["rejected_files"]), (2, 1, 1))
