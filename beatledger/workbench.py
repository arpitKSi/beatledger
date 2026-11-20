"""Editor review rules, source-backed revisions, and the bundled example."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from run_batch import BatchAnalysis, evidence_record, serialize_segments

from .parser import Episode
from .ranking import BaselineScorer, Pacing
from .verifier import CutError, Segment, SegmentSpec, Timeline, verify_cut


SAMPLE_FILES = (
    "tests/fixtures/stitches/episode-101.txt",
    "tests/fixtures/stitches/episode-102.txt",
    "tests/fixtures/workbench/episode-103.txt",
)
SAVED_SAMPLE = "examples/judged_sample.json"
UNJUDGED = "Unjudged"
DecisionStatus = Literal["pending", "approved", "rejected"]


def sample_sources(project_root: Path) -> list[tuple[str, bytes]]:
    paths = [project_root / relative for relative in SAMPLE_FILES]
    # Give the bundled inputs the same bytes on Git checkouts with either line ending convention.
    return [(path.name, path.read_text(encoding="utf-8").encode("utf-8")) for path in paths]


def load_saved_sample(project_root: Path, baseline: BatchAnalysis) -> BatchAnalysis:
    payload = json.loads((project_root / SAVED_SAMPLE).read_text(encoding="utf-8"))
    records = payload["records"]
    current = {record["candidate_id"]: record for record in baseline.records}
    if len(records) != len(current):
        raise ValueError("Saved sample is out of date. Run the sample to generate a fresh list.")
    for record in records:
        original = current.pop(record["candidate_id"], None)
        if original is None or any(
            original[field] != record[field]
            for field in ("segments", "baseline_rank", "baseline_score", "dialogue_wps", "action_beat_ratio")
        ):
            raise ValueError("Saved sample is out of date. Run the sample to generate a fresh list.")
    return BatchAnalysis(baseline.episodes, records, payload["ledger"], payload["audit"])


@dataclass(frozen=True)
class ReviewCut:
    segments: tuple[Segment, ...]
    pacing: Pacing
    hook: dict | None
    cliffhanger: dict | None
    flags: tuple[str, ...]
    revised: bool = False

    @property
    def version(self) -> str:
        return "+".join(
            f"{segment.spec.episode_number}:{segment.start.id}:{segment.end.id}" for segment in self.segments
        )

    @property
    def footage_required(self) -> bool:
        return "footage_verification_required" in self.pacing.flags


@dataclass(frozen=True)
class Decision:
    status: DecisionStatus
    note: str
    footage_verified: bool
    cut_version: str


def record_specs(record: dict) -> tuple[SegmentSpec, ...]:
    return tuple(
        SegmentSpec(part["episode"], part["start_beat_id"], part["end_beat_id"])
        for part in record["segments"]
    )


def original_cut(record: dict, timelines: dict[int, Timeline]) -> ReviewCut:
    segments = verify_cut(timelines, record_specs(record))
    pacing = BaselineScorer(beat for part in segments for beat in part.beats).measure_pacing(segments)
    return ReviewCut(segments, pacing, record["hook"], record["cliffhanger"], tuple(record["review_flags"]))


def trim_cut(
    record: dict, specs: tuple[SegmentSpec, ...], timelines: dict[int, Timeline],
    recap_resume_cs: int | None = None,
) -> ReviewCut:
    original = record_specs(record)
    if specs == original:
        return original_cut(record, timelines)
    if tuple(spec.episode_number for spec in specs) != tuple(spec.episode_number for spec in original):
        raise CutError("trim_episode_order", "A trim must keep the original episodes and segment order.")
    segments = verify_cut(timelines, specs)
    scorer = BaselineScorer(beat for part in segments for beat in part.beats)
    candidate = scorer.score(segments)
    if candidate is None:
        pacing = scorer.measure_pacing(segments)
        flags = (*pacing.flags, "missing_trim_evidence")
        hook, ending = None, None
    else:
        pacing = candidate.pacing
        flags = candidate.flags
        hook, ending = evidence_record(candidate.hook), evidence_record(candidate.cliffhanger)
    if len(segments) == 2:
        recap_warning = (
            "recap_review_required" in record["review_flags"]
            or recap_resume_cs is not None and segments[1].start_cs < recap_resume_cs
        )
        flags = (*flags, *(("recap_review_required",) if recap_warning else ()))
    return ReviewCut(segments, pacing, hook, ending, (*dict.fromkeys(flags), "edited_cut"), revised=True)


def review_decision(cut: ReviewCut, status: DecisionStatus, note: str, footage_verified: bool) -> Decision:
    if status == "approved" and cut.footage_required and not footage_verified:
        raise ValueError("Confirm that you checked the footage before approving this zero-dialogue cut.")
    return Decision(status, note.strip(), footage_verified, cut.version)


def filter_records(
    records: list[dict], decisions: dict[str, Decision], *, archetypes: list[str],
    tension: tuple[int, int], pacing: tuple[float, float], status: str,
) -> list[dict]:
    visible = []
    for record in records:
        if archetypes and (record["hook_archetype"] or UNJUDGED) not in archetypes:
            continue
        delta = record["tension_delta"]
        if tension != (-10, 10) and (delta is None or not tension[0] <= delta <= tension[1]):
            continue
        if not pacing[0] <= record["dialogue_wps"] <= pacing[1]:
            continue
        decision = decisions.get(record["candidate_id"])
        if status != "all" and (decision.status if decision else "pending") != status:
            continue
        visible.append(record)
    return visible


def decision_export(
    records: list[dict], episodes: dict[int, Episode], timelines: dict[int, Timeline],
    revisions: dict[str, ReviewCut], decisions: dict[str, Decision],
) -> dict:
    reviews = []
    for record in records:
        candidate_id = record["candidate_id"]
        revision, decision = revisions.get(candidate_id), decisions.get(candidate_id)
        if revision is None and decision is None:
            continue
        cut = revision or original_cut(record, timelines)
        if decision is not None and decision.cut_version != cut.version:
            decision = None
        reviews.append({
            "candidate_id": candidate_id,
            "original_rank": record["rank"],
            "status": decision.status if decision else "pending",
            "note": decision.note if decision else "",
            "revised": cut.revised,
            "footage_verification_required": cut.footage_required,
            "footage_verified": decision.footage_verified if decision else False,
            "segments": serialize_segments(cut.segments, episodes),
            "original_segments": record["segments"],
            "duration_cs": cut.pacing.duration_cs,
            "dialogue_wps": cut.pacing.dialogue_wps,
            "action_beat_ratio": cut.pacing.action_beat_ratio,
            "hook_archetype": record["hook_archetype"] if not cut.revised else None,
            "opening_tension": record["opening_tension"] if not cut.revised else None,
            "ending_tension": record["ending_tension"] if not cut.revised else None,
            "tension_delta": record["tension_delta"] if not cut.revised else None,
            "hook": cut.hook,
            "cliffhanger": cut.cliffhanger,
            "review_flags": list(cut.flags),
        })
    return {"reviews": reviews}
