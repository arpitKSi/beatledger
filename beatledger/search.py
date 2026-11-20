"""Shortlist continuous cuts and chronological stitches for a screenplay pack."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Iterable
from dataclasses import dataclass

from .parser import Episode
from .ranking import BaselineScorer, Candidate, candidate_order
from .recap import RecapCheck, inspect_recap
from .verifier import MAX_DURATION_CS, MIN_DURATION_CS, Segment, SegmentSpec, Timeline, verify_cut


CONTINUOUS_LIMIT = 3
DUPLICATE_OVERLAP_PERCENT = 75
MIN_STITCH_PART_CS = 800
MAX_STITCH_PART_CS = 3_000


@dataclass(frozen=True)
class Duplicate:
    candidate_id: str
    kept_id: str
    score: float
    overlap_ratio: float
    reason: str


@dataclass(frozen=True)
class EpisodeSearch:
    episode_number: int
    scanned_windows: int
    eligible_windows: int
    selected_ids: tuple[str, ...]
    suppressed_duplicates: tuple[Duplicate, ...]


@dataclass(frozen=True)
class StitchSearch:
    episodes: tuple[int, int]
    recap: RecapCheck
    original_start_beat_id: str
    revised_start_beat_id: str | None
    selected_id: str | None
    reason: str


@dataclass(frozen=True)
class SearchResult:
    candidates: tuple[Candidate, ...]
    episodes: tuple[EpisodeSearch, ...]
    stitches: tuple[StitchSearch, ...]


def _clock_range(clocks: list[int], minimum: int, maximum: int) -> list[int]:
    return clocks[bisect_left(clocks, minimum):bisect_right(clocks, maximum)]


def _intersection_union(left: Segment, right: Segment) -> tuple[int, int]:
    return (
        max(0, min(left.end_cs, right.end_cs) - max(left.start_cs, right.start_cs)),
        max(left.end_cs, right.end_cs) - min(left.start_cs, right.start_cs),
    )


def _continuous(timeline: Timeline, scorer: BaselineScorer) -> tuple[list[Candidate], EpisodeSearch]:
    number = timeline.episode.number
    ends = sorted(timeline.ends)
    scanned = 0
    eligible = []
    for start_cs, start_id in sorted(timeline.starts.items()):
        for end_cs in _clock_range(ends, start_cs + MIN_DURATION_CS, start_cs + MAX_DURATION_CS):
            spec = SegmentSpec(number, start_id, timeline.ends[end_cs])
            segments = verify_cut({number: timeline}, (spec,))
            scanned += 1
            candidate = scorer.score(segments)
            if candidate is not None:
                eligible.append(candidate)

    selected = []
    suppressed = []
    for candidate in sorted(eligible, key=candidate_order):
        for kept in selected:
            overlap, union = _intersection_union(candidate.segments[0], kept.segments[0])
            reused_hook = candidate.hook.id == kept.hook.id
            if reused_hook or overlap * 100 >= DUPLICATE_OVERLAP_PERCENT * union:
                suppressed.append(Duplicate(
                    candidate.id, kept.id, candidate.score.total, round(overlap / union, 4),
                    "shared_hook" if reused_hook else "temporal_overlap",
                ))
                break
        else:
            selected.append(candidate)
            if len(selected) == CONTINUOUS_LIMIT:
                break
    return selected, EpisodeSearch(
        number, scanned, len(eligible), tuple(candidate.id for candidate in selected), tuple(suppressed),
    )


def _stitch(
    previous: Timeline, following: Timeline, scorer: BaselineScorer,
) -> tuple[Candidate | None, StitchSearch]:
    pair = (previous.episode.number, following.episode.number)
    timelines = {pair[0]: previous, pair[1]: following}
    recap = inspect_recap(previous.episode, following.episode)
    original_clock = min(following.starts)
    original_id = following.starts[original_clock]
    if recap.status == "confirmed":
        next_clock = next(
            (clock for clock in sorted(following.starts) if recap.resume_cs is not None and clock >= recap.resume_cs),
            None,
        )
        if next_clock is None:
            return None, StitchSearch(pair, recap, original_id, None, None, "no_safe_start_after_recap")
    else:
        next_clock = original_clock
    next_id = following.starts[next_clock]

    ending_clock = max(previous.ends)
    starts = _clock_range(
        sorted(previous.starts), ending_clock - MAX_STITCH_PART_CS, ending_clock - MIN_STITCH_PART_CS,
    )
    ends = _clock_range(
        sorted(following.ends), next_clock + MIN_STITCH_PART_CS, next_clock + MAX_STITCH_PART_CS,
    )
    first_parts = [
        previous.resolve(SegmentSpec(pair[0], previous.starts[clock], previous.ends[ending_clock]))
        for clock in starts
    ]
    second_parts = [
        following.resolve(SegmentSpec(pair[1], next_id, following.ends[clock])) for clock in ends
    ]

    first_speakers = [_speakers(part, scorer) for part in first_parts]
    second_speakers = [_speakers(part, scorer) for part in second_parts]
    options = []
    duration_ok = False
    joined = False
    flags = ("recap_review_required",) if recap.status == "ambiguous" else ()
    for first, left_speakers in zip(first_parts, first_speakers):
        for second, right_speakers in zip(second_parts, second_speakers):
            if not MIN_DURATION_CS <= first.duration_cs + second.duration_cs <= MAX_DURATION_CS:
                continue
            duration_ok = True
            if not left_speakers & right_speakers:
                continue
            joined = True
            verified = verify_cut(timelines, (first.spec, second.spec))
            candidate = scorer.score(verified, flags)
            if candidate is not None:
                options.append(candidate)

    if options:
        candidate = min(options, key=candidate_order)
        return candidate, StitchSearch(pair, recap, original_id, next_id, candidate.id, "selected")
    reason = "no_valid_duration" if not duration_ok else "no_shared_speaker" if not joined else "no_edge_evidence"
    return None, StitchSearch(pair, recap, original_id, next_id, None, reason)


def _speakers(segment: Segment, scorer: BaselineScorer) -> set[str]:
    return {
        signal.speaker for beat in segment.beats if (signal := scorer.signals[beat.id]).speaker
    }


def search_pack(episodes: Iterable[Episode]) -> SearchResult:
    timelines = {episode.number: Timeline(episode) for episode in sorted(episodes, key=lambda item: item.number)}
    scorer = BaselineScorer(beat for timeline in timelines.values() for beat in timeline.episode.beats)
    candidates = []
    coverage = []
    stitches = []
    for number, timeline in timelines.items():
        continuous, scan = _continuous(timeline, scorer)
        candidates.extend(continuous)
        coverage.append(scan)
        if number + 1 in timelines:
            stitched, audit = _stitch(timeline, timelines[number + 1], scorer)
            stitches.append(audit)
            if stitched is not None:
                candidates.append(stitched)
    return SearchResult(tuple(sorted(candidates, key=candidate_order)), tuple(coverage), tuple(stitches))
