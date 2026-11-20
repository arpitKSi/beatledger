import hashlib
import unittest
from pathlib import Path

from beatledger.parser import format_clock, parse_clock, parse_episode, parse_pack


FIXTURE = Path(__file__).parent / "fixtures" / "episode-017.txt"


class ClockTests(unittest.TestCase):
    def test_exact_centisecond_conversion_and_round_trip(self):
        for clock, centiseconds in (
            ("00:00.00", 0), ("00:59.99", 5999), ("01:00.00", 6000), ("123:04.05", 738405),
        ):
            with self.subTest(clock=clock):
                self.assertEqual(parse_clock(clock), centiseconds)
                self.assertEqual(format_clock(centiseconds), clock)

    def test_rejects_out_of_range_seconds_and_other_precisions(self):
        for clock in ("00:60.00", "00:03.1", "00:03.123", "01:01:00.00", "-00:01.00"):
            with self.subTest(clock=clock), self.assertRaises(ValueError):
                parse_clock(clock)


class EpisodeTests(unittest.TestCase):
    def setUp(self):
        self.source = FIXTURE.read_bytes()

    def test_retains_scene_metadata_and_ignores_material_outside_script(self):
        result = parse_episode("arbitrary-filename.txt", self.source)
        self.assertFalse(result.issues)
        episode = result.episode
        self.assertEqual(episode.number, 17)
        self.assertEqual(episode.filename, "arbitrary-filename.txt")
        self.assertEqual(episode.source_sha256, hashlib.sha256(self.source).hexdigest())
        self.assertEqual(len(episode.beats), 5)
        self.assertEqual(episode.scenes[0].heading, "INT. OFFICE - RECAP")
        self.assertEqual((episode.scenes[0].start_cs, episode.scenes[0].end_cs), (0, 1510))

    def test_overlaps_timeline_sort_source_ids_and_multiline_text(self):
        episode = parse_episode(FIXTURE.name, self.source).episode
        self.assertEqual(
            [(beat.id, beat.start_cs, beat.end_cs) for beat in episode.beats],
            [
                ("E017-B001", 0, 512),
                ("E017-B003", 350, 615),
                ("E017-B002", 403, 1005),
                ("E017-B004", 1510, 1540),
                ("E017-B005", 1510, 1701),
            ],
        )
        dialogue = episode.beats[1]
        self.assertEqual(dialogue.text, '"Why are there two keys?\nWho sent the second one?"')
        self.assertEqual((dialogue.kind, dialogue.scene_number), ("dialogue", 1))
        lines = self.source.decode("utf-8").splitlines()
        self.assertEqual(lines[dialogue.source_line - 1], "**NORA (Quiet):**")
        self.assertEqual(lines[dialogue.timestamp_line - 1], "Start: (00:03.50) | End: (00:06.15)")
        self.assertEqual([beat.kind for beat in episode.beats], ["camera", "dialogue", "action", "production", "dialogue"])

    def test_accepts_a_bom_crlf_and_case_insensitive_clock_labels(self):
        source = self.source.decode().replace("Start:", "START:").replace("End:", "END:")
        source = ("\ufeff" + "\r\n".join(source.splitlines())).encode("utf-8")
        result = parse_episode(FIXTURE.name, source)
        self.assertFalse(result.issues)
        self.assertEqual(len(result.episode.beats), 5)
        self.assertEqual(result.episode.beats[1].timestamp_line, 24)

    def test_skips_malformed_beat_and_retains_later_source_ordinal(self):
        source = self.source.replace(b"00:04.03", b"00:99.03")
        result = parse_episode(FIXTURE.name, source)
        self.assertEqual(len(result.episode.beats), 4)
        self.assertEqual(result.episode.beats[1].id, "E017-B003")
        self.assertEqual(
            [(issue.code, issue.line, issue.severity) for issue in result.issues],
            [("invalid_beat_clock", 20, "error")],
        )

    def test_flags_source_scene_mismatch_without_changing_beat_clock(self):
        source = self.source.replace(b"00:42.00", b"00:16.00")
        result = parse_episode(FIXTURE.name, source)
        beat = result.episode.beats[-1]
        self.assertEqual(beat.end_cs, 1701)
        self.assertEqual(result.episode.scenes[-1].end_cs, 1600)
        self.assertEqual(
            [(issue.code, issue.line, issue.severity) for issue in result.issues],
            [("beat_outside_scene_clock", beat.timestamp_line, "warning")],
        )

    def test_inverted_beat_clock_is_rejected(self):
        source = self.source.replace(b"End: (00:05.12)", b"End: (00:00.00)")
        result = parse_episode(FIXTURE.name, source)
        self.assertEqual(len(result.episode.beats), 4)
        self.assertEqual(result.issues[0].code, "inverted_beat_clock")

    def test_orphan_clock_is_reported_and_kept_out_of_previous_beat_text(self):
        source = self.source.replace(
            b"SCENE 02", b"Start: (00:11.00) | End: (00:12.00)\nUnlabeled material.\n\nSCENE 02",
        )
        result = parse_episode(FIXTURE.name, source)
        self.assertEqual(result.issues[0].code, "orphan_timestamps")
        self.assertEqual(result.episode.beats[1].text, '"Why are there two keys?\nWho sent the second one?"')

    def test_allows_an_undelimited_episode_without_scene_labels(self):
        source = b"EPISODE NUMBER: 8\n**ACTION:**\nStart: (00:00.00) | End: (00:00.01)\nA door shuts."
        result = parse_episode("new.txt", source)
        self.assertFalse(result.issues)
        self.assertEqual(result.episode.scenes, ())
        self.assertIsNone(result.episode.beats[0].scene_number)

    def test_rejects_invalid_encoding_missing_header_and_missing_beats(self):
        for content, code in (
            (b"\xff", "invalid_encoding"),
            (b"No episode number", "invalid_episode_header"),
            (b"Episode number: 4\n**ACTION:**", "no_valid_beats"),
        ):
            with self.subTest(code=code):
                result = parse_episode("invalid.txt", content)
                self.assertIsNone(result.episode)
                self.assertIn(code, {issue.code for issue in result.issues})

    def test_duplicate_ids_reject_both_files_without_discarding_other_episodes(self):
        different = self.source.replace(b"Episode 17", b"Episode 18", 1)
        first, second, third = parse_pack(
            (("one.txt", self.source), ("two.txt", different), ("three.txt", self.source)),
        )
        for result in (first, third):
            self.assertIsNone(result.episode)
            self.assertEqual(result.episode_number, 17)
            self.assertEqual(result.issues[-1].code, "duplicate_episode_number")
        self.assertEqual(second.episode.number, 18)
