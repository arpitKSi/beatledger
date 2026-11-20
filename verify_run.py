"""Independently check an exported run against its original screenplay files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path


_TIME = re.compile(r"Start:\s*\(([0-9]+:[0-5][0-9]\.[0-9]{2})\)\s*\|\s*End:\s*\(([0-9]+:[0-5][0-9]\.[0-9]{2})\)", re.I)
_LABEL = re.compile(r"\*\*(.+):\*\*")
_SECTION = re.compile(r"Scene(?:\s+[0-9]+|\s+(?:heading|start|end):.*)", re.I)
_PRODUCTION = {"sfx", "sound fx", "sound effects", "music", "transition"}
_WORD = re.compile(r"[^\W_]+(?:[’'-][^\W_]+)*")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def centiseconds(text: str) -> int:
    minutes, seconds, fraction = re.split(r"[:.]", text)
    return (int(minutes) * 60 + int(seconds)) * 100 + int(fraction)


@dataclass(frozen=True)
class SourceBeat:
    label_line: int
    time_line: int
    start: int
    end: int
    text: str
    kind: str


def read_source(content: bytes, episode: int) -> dict[str, SourceBeat]:
    lines = [line.strip() for line in content.decode("utf-8-sig").splitlines()]
    lower = [line.casefold() for line in lines]
    start = lower.index("--- script start ---") + 1 if "--- script start ---" in lower else 0
    end = lower.index("--- script end ---") if "--- script end ---" in lower else len(lines)
    headers = [
        int(match[1]) for line in lines[:start or end]
        if (match := re.fullmatch(r"Episode number:\s*(?:Episode\s+)?([0-9]+)", line, re.I))
    ]
    require(headers == [episode], f"Episode header disagrees with the ledger for episode {episode}.")
    blocks = [
        index for index in range(start, end)
        if _LABEL.fullmatch(lines[index]) or _SECTION.fullmatch(lines[index])
    ]
    beats = {}
    for label_row, next_block in zip(blocks, (*blocks[1:], end)):
        label = _LABEL.fullmatch(lines[label_row])
        if not label:
            continue
        clock_row = next((index for index in range(label_row + 1, next_block) if lines[index]), None)
        timing = _TIME.fullmatch(lines[clock_row]) if clock_row is not None else None
        require(timing is not None, f"Missing source clock in episode {episode}, line {label_row + 1}.")
        start_cs, end_cs = map(centiseconds, timing.groups())
        name = label[1].casefold().strip()
        kind = "production" if name.partition("(")[0].strip() in _PRODUCTION else name
        if kind not in {"production", "action", "camera"}:
            kind = "dialogue"
        body = " ".join(" ".join(lines[clock_row + 1:next_block]).split())
        require(start_cs < end_cs and bool(body), f"Invalid source beat in episode {episode}, line {label_row + 1}.")
        beats[f"E{episode:03d}-B{len(beats) + 1:03d}"] = SourceBeat(
            label_row + 1, clock_row + 1, start_cs, end_cs, body, kind,
        )
    return beats


def verify(input_dir: Path, run: Path, expected_episodes: int | None) -> dict[str, int]:
    ledger = json.loads((run / "run_ledger.json").read_text(encoding="utf-8"))
    records = json.loads((run / "ranked_promos.json").read_text(encoding="utf-8"))["promos"]
    with (run / "ranked_promos.csv").open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    sources = {path.name: path.read_bytes() for path in input_dir.iterdir() if path.suffix.casefold() == ".txt"}
    files = ledger["inputs"]["files"]
    require(ledger["inputs"]["summary"]["errors"] == 0, "The run ledger contains source errors.")
    require(len(files) == len(sources) and {row["filename"] for row in files} == sources.keys(), "Input coverage differs.")
    require(expected_episodes is None or len(sources) == expected_episodes, "Unexpected episode count.")
    timelines = {}
    for item in files:
        filename, episode = item["filename"], item["episode_number"]
        digest = hashlib.sha256(sources[filename]).hexdigest()
        require(item["source_sha256"] == digest, f"Source digest disagrees for episode {episode}.")
        beats = read_source(sources[filename], episode)
        require(len(beats) == item["beats"], f"Beat count disagrees for episode {episode}.")
        timelines[episode] = (filename, digest, beats)

    require(len(records) == len(rows) == len({record["candidate_id"] for record in records}), "Rank lists differ.")
    covered = set()
    checked_segments = 0
    for rank, (record, row) in enumerate(zip(records, rows), 1):
        cut = record["candidate_id"]
        require(record["rank"] == rank and row["rank"] == str(rank) and row["candidate_id"] == cut, "Rank order differs.")
        parts = record["segments"]
        order = [part["episode"] for part in parts]
        require(order == record["episodes"] and len(parts) in {1, 2}, f"{cut}: segment count or episodes differ.")
        if len(parts) == 2:
            require(order[1] == order[0] + 1, f"{cut}: stitched episodes are not adjacent and ordered.")
        total = 0
        included = {}
        for part in parts:
            filename, digest, beats = timelines[part["episode"]]
            start, end = part["start_cs"], part["end_cs"]
            require(part["file"] == filename and part["source_sha256"] == digest, f"{cut}: segment provenance differs.")
            for side in ("start", "end"):
                edge = beats[part[f"{side}_beat_id"]]
                require(
                    getattr(edge, side) == part[f"{side}_cs"] == centiseconds(part[side])
                    and edge.time_line == part[f"{side}_timestamp_line"],
                    f"{cut}: {side} disagrees with its recorded source line.",
                )
            require(start < end and not any(
                beat.kind == "dialogue" and (beat.start < start < beat.end or beat.start < end < beat.end)
                for beat in beats.values()
            ), f"{cut}: invalid segment or a dialogue split.")
            included.update((key, beat) for key, beat in beats.items() if beat.start < end and start < beat.end)
            total += end - start
        require(2000 <= total <= 6000 and record["duration_cs"] == total, f"{cut}: invalid total duration.")
        for field, part in (("hook", parts[0]), ("cliffhanger", parts[-1])):
            evidence = record[field]
            beat = timelines[part["episode"]][2][evidence["beat_id"]]
            require(
                beat.kind != "production" and beat.text == evidence["text"]
                and beat.label_line == evidence["source_line"] and beat.time_line == evidence["timestamp_line"]
                and beat.start == centiseconds(evidence["start"]) and beat.end == centiseconds(evidence["end"])
                and part["start_cs"] <= beat.start < beat.end <= part["end_cs"]
                and (beat.start < part["start_cs"] + 300 if field == "hook" else beat.end >= part["end_cs"] - 500)
                and row[f"{field}_beat_id"] == evidence["beat_id"],
                f"{cut}: {field} differs or lies outside its evidence window.",
            )
        require(
            row["segments"] == " | ".join(f'E{part["episode"]:03d} {part["start"]}–{part["end"]}' for part in parts)
            and float(row["duration_seconds"]) == record["duration_seconds"] == total / 100
            and float(row["dialogue_wps"]) == record["dialogue_wps"]
            and float(row["action_beat_ratio"]) == record["action_beat_ratio"],
            f"{cut}: CSV and JSON differ.",
        )
        dialogue = [beat for beat in included.values() if beat.kind == "dialogue"]
        words = sum(len(_WORD.findall(beat.text)) for beat in dialogue)
        non_dialogue = len(included) - len(dialogue)
        require(
            record["spoken_words"] == words and record["beat_count"] == len(included)
            and record["non_dialogue_count"] == non_dialogue
            and record["dialogue_wps"] == round(words * 100 / total, 2)
            and record["action_beat_ratio"] == non_dialogue / len(included)
            and record["footage_verification_required"] == (words == 0),
            f"{cut}: pacing does not match the included source beats.",
        )
        if record["tension_delta"] is not None:
            require(
                record["ending_tension"] - record["opening_tension"] == record["tension_delta"]
                and min(.35, .05 * max(0, record["tension_delta"])) == record["tension_bonus"]
                and all(int(row[key]) == record[key] for key in ("opening_tension", "ending_tension", "tension_delta")),
                f"{cut}: tension calculation or export differs.",
            )
        covered.update(order)
        checked_segments += len(parts)
    return {
        "input_episodes": len(sources), "episodes_with_cuts": len(covered),
        "cuts": len(records), "segments": checked_segments, "evidence_records": len(records) * 2,
    }


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--input", required=True, type=Path, help="Original screenplay directory")
    cli.add_argument("--run", required=True, type=Path, help="Exported run directory")
    cli.add_argument("--expected-episodes", type=int)
    args = cli.parse_args()
    try:
        summary = verify(args.input, args.run, args.expected_episodes)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        cli.exit(1, f"Verification failed. {exc}\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
