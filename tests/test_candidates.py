import unittest
from pathlib import Path

from beatledger.parser import parse_episode
from beatledger.ranking import BaselineScorer, Score
from beatledger.recap import inspect_recap
from beatledger.search import search_pack
from beatledger.verifier import CutError, SegmentSpec, Timeline, evidence_windows, verify_cut


FIXTURES = Path(__file__).parent / "fixtures" / "stitches"


def parse(source: bytes):
    result = parse_episode("test.txt", source)
    if result.episode is None:
        raise AssertionError(result.issues)
    return result.episode


def two_beats(number: int, end: str, opening_label: str = "ACTION", opening_text: str = "Nora drops a key."):
    return parse((
        f"Episode number: {number}\n"
        f"**{opening_label}:**\nStart: (00:00.00) | End: (00:02.00)\n{opening_text}\n"
        f"**ACTION:**\nStart: (00:10.00) | End: ({end})\nA sealed letter falls."
    ).encode())


def full_spec(episode):
    return SegmentSpec(episode.number, episode.beats[0].id, episode.beats[-1].id)


class VerificationTests(unittest.TestCase):
    def test_only_source_boundaries_with_inclusive_duration(self):
        for end in ("00:20.00", "01:00.00"):
            with self.subTest(end=end):
                episode = two_beats(1, end)
                (segment,) = verify_cut({1: Timeline(episode)}, (full_spec(episode),))
                self.assertEqual(segment.end, episode.beats[-1])
        for end in ("00:19.99", "01:00.01"):
            with self.subTest(end=end), self.assertRaisesRegex(CutError, "20 to 60"):
                episode = two_beats(1, end)
                verify_cut({1: Timeline(episode)}, (full_spec(episode),))

    def test_unknown_reference_and_reversed_cross_episode_cut(self):
        first = two_beats(1, "00:20.00")
        last = two_beats(2, "00:20.00")
        timelines = {1: Timeline(first), 2: Timeline(last)}
        with self.assertRaisesRegex(CutError, "boundary IDs"):
            verify_cut(timelines, (SegmentSpec(1, last.beats[0].id, first.beats[-1].id),))
        with self.assertRaisesRegex(CutError, "chronological"):
            verify_cut(timelines, (full_spec(last), full_spec(first)))
        with self.assertRaisesRegex(CutError, "nonoverlapping"):
            verify_cut(timelines, (full_spec(first), full_spec(first)))

    def test_edges_cannot_split_dialogue_on_overlapping_tracks(self):
        episode = parse(
            b"Episode number: 1\n**NORA:**\nStart: (00:00.00) | End: (00:08.00)\nI'm waiting.\n"
            b"**CAMERA:**\nStart: (00:02.00) | End: (00:05.00)\nThe door fills the frame.\n"
            b"**ACTION:**\nStart: (00:08.00) | End: (00:25.00)\nA clock ticks."
        )
        first, camera, final = episode.beats
        timeline = Timeline(episode)
        for start, end in ((camera, final), (first, camera)):
            with self.subTest(start=start.id, end=end.id), self.assertRaisesRegex(CutError, "inside a dialogue"):
                timeline.resolve(SegmentSpec(1, start.id, end.id))
        self.assertEqual(verify_cut({1: timeline}, (SegmentSpec(1, first.id, final.id),))[0].duration_cs, 2500)

    def test_only_first_and_last_segments_supply_editorial_evidence(self):
        first, last = two_beats(1, "00:20.00"), two_beats(2, "00:20.00")
        segments = verify_cut({1: Timeline(first), 2: Timeline(last)}, (full_spec(first), full_spec(last)))
        hook, cliffhanger = evidence_windows(segments)
        self.assertEqual(hook, (first.beats[0],))
        self.assertEqual(cliffhanger, (last.beats[-1],))


class RankingTests(unittest.TestCase):
    def test_pacing_ratios_and_flag_thresholds(self):
        for words, wps, flags in (
            (0, 0.0, ("footage_verification_required",)),
            (84, 4.2, ()),
            (85, 4.25, ("rushed_dialogue",)),
        ):
            with self.subTest(words=words):
                episode = (
                    two_beats(1, "00:20.00", "NORA", " ".join(["word"] * words))
                    if words else two_beats(1, "00:20.00")
                )
                segments = verify_cut({1: Timeline(episode)}, (full_spec(episode),))
                candidate = BaselineScorer(episode.beats).score(segments)
                self.assertEqual(candidate.pacing.dialogue_wps, wps)
                self.assertEqual(candidate.pacing.flags, flags)
                self.assertEqual(candidate.pacing.action_beat_ratio, 0.5 if words else 1.0)
                self.assertEqual(candidate.held_for_review, words == 0)

    def test_stitched_wps_uses_the_sum_of_segment_lengths(self):
        first = two_beats(1, "00:20.00", "NORA", "Please tell me where it is.")
        last = two_beats(2, "00:20.00")
        segments = verify_cut({1: Timeline(first), 2: Timeline(last)}, (full_spec(first), full_spec(last)))
        candidate = BaselineScorer((*first.beats, *last.beats)).score(segments)
        self.assertEqual(candidate.pacing.spoken_words, 6)
        self.assertEqual(candidate.pacing.dialogue_wps, 0.15)
        self.assertEqual(candidate.pacing.action_beat_ratio, 0.75)

    def test_sound_fx_does_not_count_as_speech_or_block_an_action_boundary(self):
        episode = parse(
            b"Episode number: 1\n**SOUND FX:**\nStart: (00:00.00) | End: (00:23.00)\nA bell rings twice.\n"
            b"**ACTION:**\nStart: (00:02.00) | End: (00:04.00)\nNora drops the key.\n"
            b"**ACTION:**\nStart: (00:21.00) | End: (00:22.50)\nA locked box falls."
        )
        spec = SegmentSpec(1, episode.beats[1].id, episode.beats[2].id)
        segments = verify_cut({1: Timeline(episode)}, (spec,))
        candidate = BaselineScorer(episode.beats).score(segments)
        self.assertEqual(candidate.pacing.spoken_words, 0)
        self.assertEqual(candidate.pacing.action_beat_ratio, 1.0)
        self.assertIn("footage_verification_required", candidate.flags)
        self.assertEqual(candidate.cliffhanger.kind, "action")

    def test_a_question_followed_by_an_answer_loses_its_ending_bonus(self):
        episode = parse(
            b"Episode number: 1\n**NORA:**\nStart: (00:00.00) | End: (00:02.00)\nYou lied about the key.\n"
            b"**NORA:**\nStart: (00:16.00) | End: (00:17.00)\nThen why did they never find it?\n"
            b"**GUARD:**\nStart: (00:18.00) | End: (00:20.00)\nBecause it was under the floor."
        )
        segments = verify_cut({1: Timeline(episode)}, (full_spec(episode),))
        candidate = BaselineScorer(episode.beats).score(segments)
        self.assertEqual(candidate.cliffhanger.id, episode.beats[-1].id)
        self.assertLess(candidate.score.cliffhanger, 6)

    def test_tension_bonus_only_rewards_rising_tension_and_is_capped(self):
        for delta, bonus, total in ((None, 0, 5.5), (-3, 0, 5.5), (0, 0, 5.5), (5, .25, 5.75), (10, .35, 5.85)):
            with self.subTest(delta=delta):
                score = Score(6, 6, 6, 6, 0.5, delta)
                self.assertAlmostEqual(score.tension_bonus, bonus)
                self.assertEqual(score.total, total)


class StitchTests(unittest.TestCase):
    def setUp(self):
        self.previous = parse((FIXTURES / "episode-101.txt").read_bytes())
        self.following = (FIXTURES / "episode-102.txt").read_bytes()

    def test_labeled_and_unlabeled_recap_shift_to_same_safe_new_beat(self):
        for source in (self.following, self.following.replace(b"PREVIOUSLY ON", b"INT. HALLWAY")):
            with self.subTest(labeled=b"PREVIOUSLY ON" in source):
                following = parse(source)
                result = search_pack((self.previous, following))
                (decision,) = result.stitches
                self.assertEqual(decision.recap.status, "confirmed")
                self.assertEqual(decision.revised_start_beat_id, "E102-B003")
                stitch = next(candidate for candidate in result.candidates if candidate.id == decision.selected_id)
                self.assertEqual(stitch.segments[1].start_cs, 500)
                self.assertNotIn("E102-B001", {beat.id for beat in stitch.segments[1].beats})
                self.assertTrue(2_000 <= stitch.pacing.duration_cs <= 6_000)

    def test_one_repeated_spoken_line_is_preserved_in_a_continuation(self):
        source = self.following.replace(b"PREVIOUSLY ON", b"INT. HALLWAY").replace(
            b"Nora slides the brass key across the marble counter.", b"Nora reaches for the unopened letter.",
        )
        result = search_pack((self.previous, parse(source)))
        (decision,) = result.stitches
        self.assertEqual(decision.recap.status, "clear")
        self.assertIsNotNone(decision.selected_id)
        self.assertEqual(decision.revised_start_beat_id, decision.original_start_beat_id)

    def test_lone_repeated_visual_is_held_for_review(self):
        source = self.following.replace(b"PREVIOUSLY ON", b"INT. HALLWAY").replace(
            b"You must open the vault tonight.", b"A new guard has already arrived.",
        )
        result = search_pack((self.previous, parse(source)))
        (decision,) = result.stitches
        self.assertEqual(decision.recap.status, "ambiguous")
        stitch = next(candidate for candidate in result.candidates if candidate.id == decision.selected_id)
        self.assertTrue(stitch.held_for_review)
        self.assertIn("recap_review_required", stitch.flags)

    def test_recap_trim_that_leaves_less_than_twenty_seconds_is_skipped(self):
        previous = two_beats(101, "00:10.50", "NORA", "Please wait here.")
        end = self.following.index(b"**ACTION:**\nStart: (00:18.00)")
        following = parse(self.following[:end])
        self.assertEqual(inspect_recap(previous, following).resume_cs, 500)
        (decision,) = search_pack((previous, following)).stitches
        self.assertEqual(decision.recap.status, "confirmed")
        self.assertIsNone(decision.selected_id)
        self.assertEqual(decision.reason, "no_valid_duration")

    def test_recap_without_a_new_beat_has_no_valid_trim(self):
        following = parse(self.following.split(b"SCENE 02", 1)[0] + b"--- SCRIPT END ---\n")
        (decision,) = search_pack((self.previous, following)).stitches
        self.assertEqual(decision.reason, "no_safe_start_after_recap")
        self.assertIsNone(decision.selected_id)

    def test_continuous_shortlist_limits_and_near_duplicate_suppression(self):
        result = search_pack((self.previous, parse(self.following)))
        for episode in result.episodes:
            self.assertLessEqual(len(episode.selected_ids), 3)
            self.assertGreaterEqual(episode.scanned_windows, episode.eligible_windows)
            for duplicate in episode.suppressed_duplicates:
                if duplicate.reason == "temporal_overlap":
                    self.assertGreaterEqual(duplicate.overlap_ratio, 0.75)
                self.assertIn(duplicate.kept_id, episode.selected_ids)
        self.assertTrue(any(episode.suppressed_duplicates for episode in result.episodes))

    def test_alternate_lengths_with_the_same_hook_are_audited(self):
        episode = parse(
            b"Episode number: 1\n**NORA:**\nStart: (00:00.00) | End: (00:02.00)\nYou lied about my sister!\n"
            b"**NORA:**\nStart: (00:18.00) | End: (00:20.00)\nWho opened the vault?\n"
            b"**NORA:**\nStart: (00:39.00) | End: (00:42.00)\nWhy is my father waiting?\n"
        )
        (selection,) = search_pack((episode,)).episodes
        self.assertTrue(any(
            duplicate.reason == "shared_hook" and duplicate.overlap_ratio < 0.75
            for duplicate in selection.suppressed_duplicates
        ))
