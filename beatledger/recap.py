"""Identify repeated opening material before building an adjacent-episode stitch."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

from .parser import Beat, Episode
from .verifier import cut_beat_kind


RECAP_SCAN_CS = 3_000
_RECAP_LABEL = re.compile(r"\b(?:recap|previously on)\b", re.I)
_VISUAL_REPLAY = re.compile(
    r"\b(?:flashback|replay|reprise|recap|previously seen|same (?:shot|footage))\b", re.I,
)
_WORD = re.compile(r"[^\W_]+", re.UNICODE)


@dataclass(frozen=True)
class RecapCheck:
    status: Literal["clear", "confirmed", "ambiguous"]
    reason: str
    resume_cs: int | None = None
    evidence_ids: tuple[str, ...] = ()
    scene_line: int | None = None


def speaker_name(label: str) -> str:
    return " ".join(_WORD.findall(label.split("(", 1)[0].casefold()))


def _fingerprint(beat: Beat) -> tuple[str, ...]:
    words = tuple(_WORD.findall(unicodedata.normalize("NFKC", beat.text).casefold()))
    if not words:
        return ()
    kind = cut_beat_kind(beat)
    speaker = speaker_name(beat.label) if kind == "dialogue" else ""
    return (kind, speaker, *words)


def inspect_recap(previous: Episode, following: Episode) -> RecapCheck:
    opening_clock = following.beats[0].start_cs
    opening = tuple(beat for beat in following.beats if beat.start_cs < opening_clock + RECAP_SCAN_CS)
    for scene_index, scene in enumerate(following.scenes):
        if scene.number == opening[0].scene_number and scene.heading and _RECAP_LABEL.search(scene.heading):
            recap_scenes = {scene.number}
            last_recap_scene = scene
            for later in following.scenes[scene_index + 1:]:
                if not later.heading or not _RECAP_LABEL.search(later.heading):
                    break
                recap_scenes.add(later.number)
                last_recap_scene = later
            replay = tuple(beat for beat in following.beats if beat.scene_number in recap_scenes)
            resume = next(
                (beat.start_cs for beat in following.beats if beat.scene_number not in recap_scenes),
                None,
            )
            if resume is not None:
                resume = max(resume, last_recap_scene.end_cs or 0, *(beat.end_cs for beat in replay))
            return RecapCheck(
                "confirmed", "opening_scene_labeled_recap", resume,
                tuple(beat.id for beat in replay), scene.source_line,
            )

    tail_end = max(beat.end_cs for beat in previous.beats)
    tail = tuple(beat for beat in previous.beats if beat.end_cs > tail_end - RECAP_SCAN_CS)
    head_story = tuple(beat for beat in opening if cut_beat_kind(beat) in {"dialogue", "action"})
    tail_story = tuple(beat for beat in tail if cut_beat_kind(beat) in {"dialogue", "action"})
    head_keys = tuple(map(_fingerprint, head_story))
    tail_keys = tuple(map(_fingerprint, tail_story))
    best_offset, best_length = 0, 0
    for offset in range(len(tail_keys)):
        length = 0
        for old, new in zip(tail_keys[offset:], head_keys):
            if not new or old != new:
                break
            length += 1
        if length > best_length:
            best_offset, best_length = offset, length

    informative_matches = {key for key in head_keys[:best_length] if len(key) >= 6}
    if len(informative_matches) >= 2:
        replay = head_story[:best_length]
        matched_tail = tail_story[best_offset:best_offset + best_length]
        replay_end = max(beat.end_cs for beat in replay)
        replay_ids = {beat.id for beat in replay}
        next_story = next(
            (beat.start_cs for beat in following.beats if cut_beat_kind(beat) in {"dialogue", "action"}
             and beat.start_cs >= replay_end and beat.id not in replay_ids),
            None,
        )
        return RecapCheck(
            "confirmed", "ordered_opening_beats_repeat_previous_tail", next_story,
            tuple(beat.id for beat in (*matched_tail, *replay)),
        )

    visual_cues = tuple(
        beat.id for beat in opening
        if cut_beat_kind(beat) != "dialogue" and _VISUAL_REPLAY.search(beat.text)
    )
    tail_visuals = {
        key: beat.id for beat in tail
        if beat.kind in {"action", "camera"} and len(key := _fingerprint(beat)) >= 6
    }
    repeated_visuals = tuple(
        reference
        for beat in opening if beat.kind in {"action", "camera"}
        if (match := tail_visuals.get(_fingerprint(beat))) is not None
        for reference in (match, beat.id)
    )
    if visual_cues or repeated_visuals:
        return RecapCheck(
            "ambiguous", "possible_visual_replay", evidence_ids=tuple(dict.fromkeys((*visual_cues, *repeated_visuals))),
        )
    return RecapCheck("clear", "no_corroborated_opening_replay")
