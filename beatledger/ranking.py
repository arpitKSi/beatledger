"""Measure pacing and rank verified candidates with an offline editorial baseline."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from .parser import Beat, BeatKind
from .recap import speaker_name
from .verifier import HOOK_WINDOW_CS, Segment, cut_beat_kind, evidence_windows


_WORDS = re.compile(r"[^\W_]+(?:[’'-][^\W_]+)*")
_CUES = (
    ("threat", re.compile(r"\b(?:kill\w*|poison\w*|blood|dead|die|weapon|sword|gun|attack\w*|stab\w*)\b", re.I)),
    ("confrontation", re.compile(r"\b(?:slap\w*|strike\w*|struck|grabs?|punch\w*|shove\w*|lung\w*|scream\w*)\b", re.I)),
    ("reveal or accusation", re.compile(r"\b(?:truth|secret\w*|betray\w*|lied|liar|guilty|confess\w*|witness\w*|impostor)\b", re.I)),
    ("demand or consequence", re.compile(r"\b(?:choose|either|unless|or else|must|never|refuse\w*|promise\w*|deadline)\b", re.I)),
)
_CONTEXT = re.compile(
    r"\b(?:husband|wife|fianc[eé]\w*|daughter|son|brother|sister|mother|father|"
    r"wedding|marry|marriage|divorce|debt|inherit\w*|prison|trial|betray\w*)\b", re.I,
)
_INTENT = re.compile(r"\b(?:because|unless|must|want to|need to|going to|if you)\b", re.I)
_RESOLUTION = re.compile(r"\b(?:it is over|it's over|all over|we are safe|we're safe|case closed)\b", re.I)
_CONTEXT_DEPENDENT_OPENING = re.compile(r"""^[\s"'“‘]*(?:because|then (?:why|how)|and (?:when|why))\b""", re.I)


@dataclass(frozen=True)
class Pacing:
    duration_cs: int
    spoken_words: int
    beat_count: int
    non_dialogue_count: int
    dialogue_wps: float
    action_beat_ratio: float
    flags: tuple[str, ...]


@dataclass(frozen=True)
class Score:
    hook: int
    cliffhanger: int
    clarity: int
    drama: int
    pacing_penalty: float
    tension_delta: int | None = None

    @property
    def tension_bonus(self) -> float:
        return min(0.35, 0.05 * max(0, self.tension_delta or 0))

    @property
    def total(self) -> float:
        return round(
            0.35 * self.hook + 0.30 * self.cliffhanger + 0.25 * self.clarity + 0.10 * self.drama
            - self.pacing_penalty + self.tension_bonus, 2,
        )


@dataclass(frozen=True)
class Candidate:
    segments: tuple[Segment, ...]
    pacing: Pacing
    score: Score
    hook: Beat
    cliffhanger: Beat
    rationale: str
    flags: tuple[str, ...]

    @property
    def id(self) -> str:
        return "+".join(
            f"E{segment.spec.episode_number:03d}:{segment.start_cs}-{segment.end_cs}"
            for segment in self.segments
        )

    @property
    def held_for_review(self) -> bool:
        return any(flag in self.flags for flag in (
            "recap_review_required", "footage_verification_required", "incoherent_stitch",
            "stitch_ending_review",
        ))


def candidate_order(candidate: Candidate) -> tuple[bool, float, int, int, int, str]:
    return (
        candidate.held_for_review, -candidate.score.total, -candidate.score.hook,
        -candidate.score.cliffhanger, len(candidate.segments), candidate.id,
    )


def timing_penalty(segments: tuple[Segment, ...], hook: Beat, ending: Beat) -> float:
    return (
        0.5 * (hook.end_cs > segments[0].start_cs + HOOK_WINDOW_CS)
        + 0.5 * (segments[-1].end_cs - ending.end_cs > 200)
    )


def context_dependent_opening(beat: Beat) -> bool:
    return cut_beat_kind(beat) == "dialogue" and bool(_CONTEXT_DEPENDENT_OPENING.match(beat.text))


@dataclass(frozen=True)
class _BeatSignals:
    kind: BeatKind
    words: int
    speaker: str
    cues: tuple[str, ...]
    question: bool
    emphasis: bool
    context: bool
    intent: bool
    resolved: bool


def _beat_signals(beat: Beat) -> _BeatSignals:
    kind = cut_beat_kind(beat)
    story = kind in {"action", "dialogue"}
    return _BeatSignals(
        kind,
        len(_WORDS.findall(beat.text)) if kind == "dialogue" else 0,
        speaker_name(beat.label) if kind == "dialogue" else "",
        tuple(label for label, pattern in _CUES if pattern.search(beat.text)) if story else (),
        kind == "dialogue" and "?" in beat.text,
        kind == "dialogue" and "!" in beat.text,
        story and bool(_CONTEXT.search(beat.text)),
        story and bool(_INTENT.search(beat.text)),
        kind == "dialogue" and bool(_RESOLUTION.search(beat.text)),
    )


class BaselineScorer:
    def __init__(self, beats: Iterable[Beat]):
        self.signals = {beat.id: _beat_signals(beat) for beat in beats}

    def measure_pacing(self, segments: tuple[Segment, ...]) -> Pacing:
        beat_ids = dict.fromkeys(beat.id for segment in segments for beat in segment.beats)
        return self._pacing(segments, [self.signals[beat_id] for beat_id in beat_ids])

    @staticmethod
    def _pacing(segments: tuple[Segment, ...], signals: list[_BeatSignals]) -> Pacing:
        duration = sum(segment.duration_cs for segment in segments)
        spoken = sum(signal.words for signal in signals)
        wps = round(spoken * 100 / duration, 2)
        non_dialogue = sum(signal.kind != "dialogue" for signal in signals)
        flags = ()
        if wps == 0:
            flags = ("footage_verification_required",)
        elif wps > 4.2:
            flags = ("rushed_dialogue",)
        return Pacing(duration, spoken, len(signals), non_dialogue, wps, non_dialogue / len(signals), flags)

    def score(self, segments: tuple[Segment, ...], extra_flags: tuple[str, ...] = ()) -> Candidate | None:
        openings, endings = evidence_windows(segments)
        if not openings or not endings:
            return None

        first, last = segments[0], segments[-1]
        hook = max(openings, key=lambda beat: self._opening(beat, first))
        last_dialogue = max(
            ((beat.start_cs, beat.source_ordinal) for beat in last.beats if self.signals[beat.id].kind == "dialogue"),
            default=(-1, -1),
        )
        cliffhanger = max(endings, key=lambda beat: self._ending(beat, last, last_dialogue))
        hook_score = self._opening(hook, first)[0]
        cliffhanger_score = self._ending(cliffhanger, last, last_dialogue)[0]
        beats = {beat.id: beat for segment in segments for beat in segment.beats}
        signals = [self.signals[beat_id] for beat_id in beats]
        speakers = {signal.speaker for signal in signals if signal.speaker}
        context_dependent = context_dependent_opening(hook)
        clarity = min(10, (
            3 + 2 * any(signal.context for signal in signals) + 2 * any(signal.intent for signal in signals)
            + int(1 <= len(speakers) <= 2) + int(len(segments) == 1) - context_dependent
        ))
        story_count = sum(signal.kind in {"action", "dialogue"} for signal in signals)
        drama = min(10, round(12 * sum(bool(s.cues) for s in signals) / max(1, story_count)))
        penalty = timing_penalty(segments, hook, cliffhanger)
        result = Score(hook_score, cliffhanger_score, clarity, drama, penalty)

        pacing = self._pacing(segments, signals)

        flags = [*pacing.flags, *extra_flags]
        if min(hook_score, cliffhanger_score, clarity) < 6:
            flags.append("low_criterion_score")
        if penalty:
            flags.append("pacing_timing_review")
        if context_dependent:
            flags.append("context_dependent_opening")
        if len(segments) == 2:
            flags.append("stitch_continuity_review")

        opening_cue = self._description(hook)
        ending_cue = self._description(cliffhanger)
        rationale = f"Opens on {opening_cue} and cuts on {ending_cue}"
        if clarity < 6:
            rationale += ", with context that may need review"
        return Candidate(segments, pacing, result, hook, cliffhanger, rationale + ".", tuple(flags))

    def _opening(self, beat: Beat, segment: Segment) -> tuple[int, int, int]:
        signal = self.signals[beat.id]
        score = 2 + 3 * bool(signal.cues) + int(len(signal.cues) > 1) + signal.question + signal.emphasis
        score += int(signal.kind in {"action", "dialogue"}) + int(beat.end_cs <= segment.start_cs + HOOK_WINDOW_CS)
        return min(10, score), -beat.start_cs, -beat.source_ordinal

    def _ending(self, beat: Beat, segment: Segment, last_dialogue: tuple[int, int]) -> tuple[int, int, int]:
        signal = self.signals[beat.id]
        terminal = (beat.start_cs, beat.source_ordinal) >= last_dialogue
        question_bonus = (2 if terminal else -2) if signal.question else 0
        score = 2 + 3 * bool(signal.cues) + question_bonus + signal.emphasis - 2 * signal.resolved
        score += int(signal.kind in {"action", "dialogue"}) + int(segment.end_cs - beat.end_cs <= 100)
        return max(0, min(10, score)), beat.end_cs, -beat.source_ordinal

    def _description(self, beat: Beat) -> str:
        signal = self.signals[beat.id]
        if signal.question:
            return "a question"
        if signal.cues:
            return f"a {signal.cues[0]} cue"
        return {"dialogue": "a line of dialogue", "action": "an action beat", "camera": "a camera cue"}[signal.kind]
