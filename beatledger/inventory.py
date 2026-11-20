"""Inspect screenplay inputs and report parser coverage without exporting script text."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from .parser import ParseIssue, ParseResult, parse_pack


def build_inventory(results: tuple[ParseResult, ...]) -> dict:
    files = []
    totals: Counter[str] = Counter()
    episode_numbers = []
    for result in results:
        episode = result.episode
        issues = Counter(issue.severity for issue in result.issues)
        beat_counts = Counter(beat.kind for beat in episode.beats) if episode else Counter()
        source_order = sorted(episode.beats, key=lambda beat: beat.source_ordinal) if episode else []
        pairs = list(zip(source_order, source_order[1:]))
        entry = {
            "filename": result.filename,
            "episode_number": result.episode_number,
            "status": "parsed" if episode else "rejected",
            "source_sha256": episode.source_sha256 if episode else None,
            "scenes": len(episode.scenes) if episode else 0,
            "beats": sum(beat_counts.values()),
            "beats_by_kind": dict(sorted(beat_counts.items())),
            "adjacent_overlaps": sum(right.start_cs < left.end_cs for left, right in pairs),
            "backward_starts": sum(right.start_cs < left.start_cs for left, right in pairs),
            "issues": [asdict(issue) for issue in result.issues],
        }
        if episode:
            episode_numbers.append(episode.number)
        totals.update({
            "files": 1,
            "parsed_files": int(episode is not None),
            "rejected_files": int(episode is None),
            "files_with_issues": int(bool(result.issues)),
            "scenes": entry["scenes"],
            "beats": entry["beats"],
            "adjacent_overlaps": entry["adjacent_overlaps"],
            "backward_starts": entry["backward_starts"],
            "warnings": issues["warning"],
            "errors": issues["error"],
        })
        files.append(entry)

    return {
        "summary": {**totals, "parsed_episode_numbers": sorted(episode_numbers)},
        "files": files,
    }


def read_pack(source: Path) -> tuple[ParseResult, ...]:
    if source.is_dir():
        paths = sorted(
            (path for path in source.iterdir() if path.suffix.lower() == ".txt" and path.is_file()),
            key=lambda path: (path.name.casefold(), path.name),
        )
    elif source.is_file() and source.suffix.lower() == ".txt":
        paths = [source]
    else:
        paths = []
    if not paths:
        raise ValueError("No screenplay .txt files found at --input.")

    sources = []
    unreadable = {}
    for path in paths:
        try:
            sources.append((path.name, path.read_bytes()))
        except OSError as exc:
            unreadable[path.name] = ParseResult(path.name, None, None, (
                ParseIssue("read_error", f"Could not read this screenplay. {exc.strerror or 'Check access.'}"),
            ))
    by_filename = {result.filename: result for result in parse_pack(sources)}
    by_filename.update(unreadable)
    return tuple(by_filename[path.name] for path in paths)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Screenplay .txt file or pack directory")
    parser.add_argument("--output", type=Path, help="New JSON inventory file. Defaults to stdout.")
    args = parser.parse_args(argv)

    try:
        results = read_pack(args.input)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    report = build_inventory(results)
    document = json.dumps(report, indent=2) + "\n"

    if args.output is None:
        sys.stdout.write(document)
    else:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8", newline="\n") as output:
                output.write(document)
        except OSError as exc:
            parser.error(f"Cannot write inventory. {exc.strerror or 'Check --output.'} Use a new path.")

    return int(report["summary"]["errors"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
