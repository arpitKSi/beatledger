"""Resolve proposed cuts against the original beat boundaries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .parser import Beat, BeatKind, Episode


MIN_DURATION_CS = 2_000
MAX_DURATION_CS = 6_000
HOOK_WINDOW_CS = 300
ENDING_WINDOW_CS = 500


@dataclass(frozen=True)
class SegmentSpec:
    episode_number: int
    start_beat_id: str
    end_beat_id: str


@dataclass(frozen=True)
class Segment:
    spec: SegmentSpec
    start: Beat
    end: Beat
    beats: tuple[Beat, ...]

    @property
    def start_cs(self) -> int:
        return self.start.start_cs

    @property
    def end_cs(self) -> int:
        return self.end.end_cs

    @property
    def duration_cs(self) -> int:
        return self.end_cs - self.start_cs


class CutError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def cut_beat_kind(beat: Beat) -> BeatKind:
    """Recognize source sound annotations when deciding what counts as speech."""
    if beat.kind == "dialogue" and beat.label.partition("(")[0].strip().casefold() in {
        "sound fx", "sound effects",
    }:
        return "production"
    return beat.kind


class Timeline:
    def __init__(self, episode: Episode):
        self.episode = episode
        self.beats = {beat.id: beat for beat in episode.beats}
        dialogue = tuple(beat for beat in episode.beats if cut_beat_kind(beat) == "dialogue")
        self.starts: dict[int, str] = {}
        self.ends: dict[int, str] = {}
        for beat in episode.beats:
            if not any(line.start_cs < beat.start_cs < line.end_cs for line in dialogue):
                self.starts.setdefault(beat.start_cs, beat.id)
            if not any(line.start_cs < beat.end_cs < line.end_cs for line in dialogue):
                self.ends.setdefault(beat.end_cs, beat.id)

    def resolve(self, spec: SegmentSpec) -> Segment:
        start = self.beats.get(spec.start_beat_id)
        end = self.beats.get(spec.end_beat_id)
        if spec.episode_number != self.episode.number or start is None or end is None:
            raise CutError("unknown_boundary", "Both boundary IDs must belong to the specified episode.")
        if start.start_cs >= end.end_cs:
            raise CutError("nonpositive_segment", "Segment end must follow its start.")
        if start.start_cs not in self.starts or end.end_cs not in self.ends:
            raise CutError("dialogue_split", "A cut edge lies inside a dialogue beat.")
        included = tuple(
            beat for beat in self.episode.beats
            if beat.start_cs < end.end_cs and start.start_cs < beat.end_cs
        )
        return Segment(spec, start, end, included)


def verify_cut(
    timelines: Mapping[int, Timeline], specs: tuple[SegmentSpec, ...],
) -> tuple[Segment, ...]:
    if not 1 <= len(specs) <= 2:
        raise CutError("segment_count", "Expected one or two segments.")

    segments = []
    for spec in specs:
        timeline = timelines.get(spec.episode_number)
        if timeline is None:
            raise CutError("unknown_episode", "The cut refers to an unavailable episode.")
        segments.append(timeline.resolve(spec))

    if len(segments) == 2:
        first, last = segments
        if first.spec.episode_number > last.spec.episode_number or (
            first.spec.episode_number == last.spec.episode_number and first.end_cs > last.start_cs
        ):
            raise CutError("segment_order", "Segments must be chronological and nonoverlapping.")

    duration = sum(segment.duration_cs for segment in segments)
    if not MIN_DURATION_CS <= duration <= MAX_DURATION_CS:
        raise CutError("duration", "Total cut duration must be 20 to 60 seconds inclusive.")
    return tuple(segments)


def evidence_windows(segments: tuple[Segment, ...]) -> tuple[tuple[Beat, ...], tuple[Beat, ...]]:
    first, last = segments[0], segments[-1]
    hook = tuple(
        beat for beat in first.beats
        if cut_beat_kind(beat) != "production" and first.start_cs <= beat.start_cs < first.start_cs + HOOK_WINDOW_CS
        and beat.end_cs <= first.end_cs
    )
    ending = tuple(
        beat for beat in last.beats
        if cut_beat_kind(beat) != "production" and last.start_cs <= beat.start_cs
        and last.end_cs - ENDING_WINDOW_CS <= beat.end_cs <= last.end_cs
    )
    return hook, ending
