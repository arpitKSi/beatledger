import unittest
from pathlib import Path

from beatledger.parser import parse_pack
from beatledger.verifier import CutError, SegmentSpec, Timeline
from beatledger.workbench import (
    UNJUDGED, decision_export, filter_records, load_saved_sample, original_cut, review_decision,
    sample_sources, trim_cut,
)
from run_batch import analyze_pack


ROOT = Path(__file__).resolve().parents[1]


class ReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analysis = analyze_pack(parse_pack(sample_sources(ROOT)))
        cls.timelines = {number: Timeline(episode) for number, episode in cls.analysis.episodes.items()}
        cls.silent = next(
            record for record in cls.analysis.records
            if record["footage_verification_required"] and record["segments"][0]["start_cs"] == 0
        )

    def test_saved_model_run_uses_the_real_synthetic_source_segments(self):
        judged = load_saved_sample(ROOT, self.analysis)
        self.assertGreater(judged.ledger["judge"]["used_for_ranking"], 0)
        self.assertEqual({row["candidate_id"] for row in judged.records}, {row["candidate_id"] for row in self.analysis.records})
        self.assertEqual(set(judged.episodes), {101, 102, 103})

    def test_archetype_tension_and_wps_filters_use_numeric_values(self):
        judged = load_saved_sample(ROOT, self.analysis)
        eligible = next(row for row in judged.records if row["tension_delta"] > 0)
        visible = filter_records(
            judged.records, {}, archetypes=[eligible["hook_archetype"]],
            tension=(1, 10), pacing=(0.01, 20.0), status="all",
        )
        self.assertIn(eligible, visible)
        self.assertTrue(all(row["tension_delta"] >= 1 and row["dialogue_wps"] >= 0.01 for row in visible))
        self.assertEqual(
            filter_records(self.analysis.records, {}, archetypes=[UNJUDGED], tension=(-10, 10), pacing=(0.0, 20.0), status="all"),
            self.analysis.records,
        )
        self.assertFalse(filter_records(
            self.analysis.records, {}, archetypes=[], tension=(1, 10), pacing=(0.0, 20.0), status="all",
        ))

    def test_zero_dialogue_requires_confirmation_only_for_approval(self):
        cut = original_cut(self.silent, self.timelines)
        self.assertTrue(cut.footage_required)
        with self.assertRaisesRegex(ValueError, "Confirm that you checked"):
            review_decision(cut, "approved", "", False)
        self.assertEqual(review_decision(cut, "rejected", "", False).status, "rejected")
        self.assertTrue(review_decision(cut, "approved", "Checked the shot.", True).footage_verified)

    def test_trim_checks_duration_and_recomputes_metrics_for_a_new_review(self):
        record, timelines = self.silent, self.timelines
        source = timelines[103]
        cut = original_cut(record, timelines)
        with self.assertRaisesRegex(CutError, "20 to 60"):
            trim_cut(record, (SegmentSpec(103, source.starts[0], source.ends[180]),), timelines)
        revision = trim_cut(record, (SegmentSpec(103, source.starts[0], source.ends[2420]),), timelines)
        self.assertTrue(revision.revised)
        self.assertEqual(revision.pacing.duration_cs, 2420)
        self.assertEqual(revision.pacing.dialogue_wps, 0.0)
        existing = review_decision(cut, "approved", "Original shot checked.", True)
        candidate_id = record["candidate_id"]
        export = decision_export(self.analysis.records, self.analysis.episodes, timelines, {candidate_id: revision}, {candidate_id: existing})
        self.assertEqual(len(export["reviews"]), 1)
        entry = export["reviews"][0]
        self.assertEqual(entry["status"], "pending")
        self.assertFalse(entry["footage_verified"])
        self.assertEqual(entry["segments"][0]["end"], "00:24.20")
        self.assertEqual(entry["original_segments"], record["segments"])

    def test_trim_reintroducing_confirmed_recap_is_flagged(self):
        stitch = next(record for record in self.analysis.records if record["episodes"] == [101, 102])
        first, second = stitch["segments"]
        revision = trim_cut(stitch, (
            SegmentSpec(101, first["start_beat_id"], first["end_beat_id"]),
            SegmentSpec(102, self.timelines[102].starts[0], second["end_beat_id"]),
        ), self.timelines, recap_resume_cs=500)
        self.assertIn("recap_review_required", revision.flags)
