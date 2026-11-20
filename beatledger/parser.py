"""Parse timestamped screenplay blocks without changing their source clocks."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Literal


_CLOCK = r"[0-9]{2,}:[0-5][0-9]\.[0-9]{2}"
_BEAT_TIME = re.compile(rf"Start:\s*\(({_CLOCK})\)\s*\|\s*End:\s*\(({_CLOCK})\)", re.I)
_TIME_PREFIX = re.compile(r"Start:", re.I)
_ORPHAN_TIME = re.compile(r"Start:\s*\(", re.I)
_EPISODE_HEADER = re.compile(r"Episode number:\s*(?:Episode\s+)?([0-9]+)", re.I)
_SCENE_HEADER = re.compile(r"Scene[ \t]+([0-9]+)", re.I)
_SCENE_FIELD = re.compile(r"Scene (heading|start|end):\s*(.*)", re.I)
_BEAT_HEADER = re.compile(r"\*\*(.+):\*\*")
_SCRIPT_START = "--- script start ---"
_SCRIPT_END = "--- script end ---"

BeatKind = Literal["dialogue", "action", "camera", "production"]
_NON_DIALOGUE_KINDS: dict[str, BeatKind] = {
    "action": "action",
    "camera": "camera",
    "sfx": "production",
    "music": "production",
    "transition": "production",
}


@dataclass(frozen=True)
class Scene:
    number: int
    source_line: int
    heading: str | None = None
    start_cs: int | None = None
    end_cs: int | None = None


@dataclass(frozen=True)
class Beat:
    episode_number: int
    source_ordinal: int
    scene_number: int | None
    source_line: int
    timestamp_line: int
    start_cs: int
    end_cs: int
    kind: BeatKind
    label: str
    text: str

    @property
    def id(self) -> str:
        return f"E{self.episode_number:03d}-B{self.source_ordinal:03d}"


@dataclass(frozen=True)
class Episode:
    number: int
    filename: str
    source_sha256: str
    scenes: tuple[Scene, ...]
    beats: tuple[Beat, ...]


@dataclass(frozen=True)
class ParseIssue:
    code: str
    message: str
    line: int | None = None
    severity: Literal["warning", "error"] = "error"


@dataclass(frozen=True)
class ParseResult:
    filename: str
    episode_number: int | None
    episode: Episode | None
    issues: tuple[ParseIssue, ...]


def parse_clock(value: str) -> int:
    if not re.fullmatch(_CLOCK, value):
        raise ValueError("Expected a clock in MM:SS.cc format")
    minutes, seconds_and_hundredths = value.split(":")
    seconds, hundredths = seconds_and_hundredths.split(".")
    return int(minutes) * 6000 + int(seconds) * 100 + int(hundredths)


def format_clock(value: int) -> str:
    if value < 0:
        raise ValueError("A source clock cannot be negative")
    minutes, remainder = divmod(value, 6000)
    seconds, hundredths = divmod(remainder, 100)
    return f"{minutes:02d}:{seconds:02d}.{hundredths:02d}"


def parse_episode(filename: str, content: bytes) -> ParseResult:
    try:
        source_lines = content.decode("utf-8-sig").splitlines()
    except UnicodeDecodeError:
        return ParseResult(filename, None, None, (
            ParseIssue("invalid_encoding", "Save the screenplay as UTF-8."),
        ))

    lines = [line.strip() for line in source_lines]
    starts = [index for index, line in enumerate(lines) if line.casefold() == _SCRIPT_START]
    ends = [index for index, line in enumerate(lines) if line.casefold() == _SCRIPT_END]
    if len(starts) > 1 or len(ends) > 1 or (starts and ends and starts[0] >= ends[0]):
        return ParseResult(filename, None, None, (
            ParseIssue("ambiguous_script_bounds", "Use at most one ordered pair of script markers."),
        ))

    script_start = starts[0] + 1 if starts else 0
    script_end = ends[0] if ends else len(lines)
    header_end = starts[0] if starts else script_end
    headers = [
        (index + 1, int(match[1]))
        for index, line in enumerate(lines[:header_end])
        if (match := _EPISODE_HEADER.fullmatch(line))
    ]
    if len(headers) != 1 or headers[0][1] < 1:
        return ParseResult(filename, None, None, (
            ParseIssue(
                "invalid_episode_header", "Use one EPISODE NUMBER header with a positive integer.",
                headers[0][0] if headers else None,
            ),
        ))

    number = headers[0][1]
    issues: list[ParseIssue] = []
    if bool(starts) != bool(ends):
        marker = starts[0] if starts else ends[0]
        issues.append(ParseIssue(
            "unpaired_script_marker", "Only one script marker is present. Review the parsed range.",
            marker + 1, "warning",
        ))

    scene_headers = [
        (index, int(match[1]))
        for index in range(script_start, script_end)
        if (match := _SCENE_HEADER.fullmatch(lines[index]))
    ]
    sections: list[tuple[int, int | None]] = [(script_start - 1, None), *scene_headers]
    scenes: list[Scene] = []
    beats: list[Beat] = []
    scene_numbers: set[int] = set()
    ordinal = 0
    for section_index, (header_index, scene_number) in enumerate(sections):
        end = sections[section_index + 1][0] if section_index + 1 < len(sections) else script_end
        start = header_index + 1
        labels = [
            (index, match[1].strip())
            for index in range(start, end)
            if (match := _BEAT_HEADER.fullmatch(lines[index]))
        ]
        if scene_number is not None:
            metadata_end = labels[0][0] if labels else end
            scene, scene_issues = _parse_scene(lines, header_index, metadata_end, scene_number)
            scenes.append(scene)
            issues.extend(scene_issues)
            if scene_number in scene_numbers:
                issues.append(ParseIssue(
                    "duplicate_scene_number", "Use a unique scene number within the episode.",
                    header_index + 1, "warning",
                ))
            scene_numbers.add(scene_number)
        else:
            scene = None

        used_time_rows: set[int] = set()
        for label_index, (index, label) in enumerate(labels):
            ordinal += 1
            beat_end = labels[label_index + 1][0] if label_index + 1 < len(labels) else end
            beat, issue, time_row = _parse_beat(
                lines, source_lines, number, ordinal, scene_number, index, label, beat_end,
            )
            if time_row is not None:
                used_time_rows.add(time_row)
            if issue is not None:
                issues.append(issue)
            if beat is None:
                continue
            beats.append(beat)
            if (
                scene is not None and scene.start_cs is not None and scene.end_cs is not None
                and scene.start_cs < scene.end_cs
                and (beat.start_cs < scene.start_cs or beat.end_cs > scene.end_cs)
            ):
                issues.append(ParseIssue(
                    "beat_outside_scene_clock",
                    f"{beat.id} is outside the declared scene clock. Source beat timestamps were retained.",
                    beat.timestamp_line, "warning",
                ))
        for index in range(start, end):
            if index not in used_time_rows and _ORPHAN_TIME.match(lines[index]):
                issues.append(ParseIssue(
                    "orphan_timestamps", "Attach the timestamp row to a beat label.", index + 1,
                ))

    if not beats:
        issues.append(ParseIssue("no_valid_beats", "Add at least one timestamped beat with text."))
        return ParseResult(filename, number, None, tuple(issues))

    episode = Episode(
        number, filename, hashlib.sha256(content).hexdigest(), tuple(scenes),
        tuple(sorted(beats, key=lambda beat: (beat.start_cs, beat.source_ordinal))),
    )
    return ParseResult(filename, number, episode, tuple(issues))


def _parse_scene(
    lines: list[str], header_index: int, end: int, number: int,
) -> tuple[Scene, list[ParseIssue]]:
    heading = None
    clocks: dict[str, int] = {}
    seen_fields: set[str] = set()
    issues = []
    for index in range(header_index + 1, end):
        match = _SCENE_FIELD.fullmatch(lines[index])
        if not match:
            continue
        field, value = match[1].casefold(), match[2]
        if field in seen_fields:
            issues.append(ParseIssue(
                "duplicate_scene_field", f"Use one scene {field} field.", index + 1, "warning",
            ))
            continue
        seen_fields.add(field)
        if field == "heading":
            heading = value or None
            continue
        try:
            clocks[field] = parse_clock(value)
        except ValueError as exc:
            issues.append(ParseIssue("invalid_scene_clock", str(exc), index + 1, "warning"))

    start_cs, end_cs = clocks.get("start"), clocks.get("end")
    if not {"start", "end"}.issubset(seen_fields):
        issues.append(ParseIssue(
            "missing_scene_clock", "Supply both scene start and end clocks.", header_index + 1, "warning",
        ))
    if start_cs is not None and end_cs is not None and start_cs >= end_cs:
        issues.append(ParseIssue(
            "inverted_scene_clock", "Scene end must follow scene start.", header_index + 1, "warning",
        ))
    return Scene(number, header_index + 1, heading, start_cs, end_cs), issues


def _parse_beat(
    lines: list[str], source_lines: list[str], number: int, ordinal: int, scene_number: int | None,
    label_index: int, label: str, end: int,
) -> tuple[Beat | None, ParseIssue | None, int | None]:
    time_index = label_index + 1
    while time_index < end and not lines[time_index]:
        time_index += 1
    timing = _BEAT_TIME.fullmatch(lines[time_index]) if time_index < end else None
    if timing is None:
        consumed = time_index if time_index < end and _TIME_PREFIX.match(lines[time_index]) else None
        return None, ParseIssue(
            "invalid_beat_clock", "Place a valid Start/End row after the beat label.",
            consumed + 1 if consumed is not None else label_index + 1,
        ), consumed

    start_cs, end_cs = map(parse_clock, timing.groups())
    if start_cs >= end_cs:
        return None, ParseIssue(
            "inverted_beat_clock", "Beat end must follow beat start.", time_index + 1,
        ), time_index

    body_end = next(
        (index for index in range(time_index + 1, end) if _ORPHAN_TIME.match(lines[index])),
        end,
    )
    text = "\n".join(source_lines[time_index + 1:body_end]).strip()
    if not label or not text:
        return None, ParseIssue(
            "empty_beat", "Supply both a label and beat text.", label_index + 1,
        ), time_index

    beat = Beat(
        episode_number=number,
        source_ordinal=ordinal,
        scene_number=scene_number,
        source_line=label_index + 1,
        timestamp_line=time_index + 1,
        start_cs=start_cs,
        end_cs=end_cs,
        kind=_NON_DIALOGUE_KINDS.get(label.casefold(), "dialogue"),
        label=label,
        text=text,
    )
    return beat, None, time_index


def parse_pack(sources: Iterable[tuple[str, bytes]]) -> tuple[ParseResult, ...]:
    results = tuple(parse_episode(filename, content) for filename, content in sources)
    counts = Counter(result.episode_number for result in results if result.episode_number is not None)
    return tuple(
        replace(result, episode=None, issues=(*result.issues, ParseIssue(
            "duplicate_episode_number",
            f"Episode {result.episode_number} appears in multiple files. Keep one version and rerun.",
        )))
        if counts[result.episode_number] > 1 else result
        for result in results
    )
