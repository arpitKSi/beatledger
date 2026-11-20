"""Run BeatLedger and save an auditable, private output bundle."""

from __future__ import annotations

import argparse
import csv
import io
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from beatledger.inventory import build_inventory, read_pack
from beatledger.parser import Beat, Episode, ParseResult, format_clock
from beatledger.search import (
    CONTINUOUS_LIMIT, DUPLICATE_OVERLAP_PERCENT, MAX_STITCH_PART_CS, MIN_STITCH_PART_CS,
    SearchResult, search_pack,
)
from beatledger.verifier import Segment, Timeline, verify_cut


if TYPE_CHECKING:
    from beatledger.judge import Evaluation
    from beatledger.judge_provider import JudgeSettings


MODE = "deterministic_baseline"
MODEL_MODE = "model_judge"
_CSV_FIELDS = (
    "rank", "baseline_rank", "model_rank", "candidate_id", "priority_group", "score_source",
    "judge_status", "score", "baseline_score", "episodes", "segments",
    "duration_seconds", "hook", "hook_beat_id", "cliffhanger", "cliffhanger_beat_id",
    "hook_summary", "cliffhanger_summary",
    "rationale", "hook_score", "cliffhanger_score", "clarity_score", "drama_score",
    "pacing_penalty", "hook_archetype", "opening_tension", "ending_tension", "tension_delta",
    "tension_bonus", "dialogue_wps", "action_beat_ratio", "footage_verification_required", "review_flags",
)


@dataclass(frozen=True)
class BatchAnalysis:
    episodes: dict[int, Episode]
    records: list[dict]
    ledger: dict
    audit: str


def serialize_segments(segments: tuple[Segment, ...], episodes: dict[int, Episode]) -> list[dict]:
    return [
        {
            "episode": segment.spec.episode_number,
            "file": episodes[segment.spec.episode_number].filename,
            "source_sha256": episodes[segment.spec.episode_number].source_sha256,
            "start": format_clock(segment.start_cs),
            "end": format_clock(segment.end_cs),
            "start_cs": segment.start_cs,
            "end_cs": segment.end_cs,
            "start_beat_id": segment.start.id,
            "end_beat_id": segment.end.id,
            "start_timestamp_line": segment.start.timestamp_line,
            "end_timestamp_line": segment.end.timestamp_line,
        }
        for segment in segments
    ]


def evidence_record(beat: Beat) -> dict:
    return {
        "beat_id": beat.id,
        "text": " ".join(beat.text.split()),
        "source_line": beat.source_line,
        "timestamp_line": beat.timestamp_line,
        "start": format_clock(beat.start_cs),
        "end": format_clock(beat.end_cs),
    }


def promo_records(
    result: SearchResult, episodes: dict[int, Episode], evaluations: dict[str, Evaluation] | None = None,
) -> list[dict]:
    timelines = {number: Timeline(episode) for number, episode in episodes.items()}
    candidates = result.candidates
    baseline_ranks = {candidate.id: rank for rank, candidate in enumerate(candidates, 1)}
    model_rank = 0
    if evaluations is not None:
        from beatledger.judge import judged_order

        candidates = tuple(sorted(candidates, key=lambda item: judged_order(item, evaluations.get(item.id))))

    records = []
    for rank, baseline in enumerate(candidates, 1):
        evaluation = evaluations.get(baseline.id) if evaluations is not None else None
        response = evaluation.response if evaluation and evaluation.usable else None
        candidate = evaluation.candidate if evaluation else baseline
        if response is not None:
            model_rank += 1
            status = "validated"
        elif evaluation is not None:
            status = "baseline_fallback"
        else:
            status = "baseline_not_sampled" if evaluations is not None else "disabled"

        # Re-resolve IDs from source instead of trusting cached clocks at export.
        resolved = verify_cut(timelines, tuple(segment.spec for segment in candidate.segments))
        pacing = candidate.pacing
        score = candidate.score
        records.append({
            "rank": rank,
            "baseline_rank": baseline_ranks[baseline.id],
            "model_rank": model_rank if response is not None else None,
            "candidate_id": candidate.id,
            "priority_group": "review_hold" if candidate.held_for_review else "shortlist",
            "score_source": MODEL_MODE if response is not None else MODE,
            "judge_status": status,
            "judge_error": evaluation.error if evaluation else None,
            "score": score.total,
            "baseline_score": baseline.score.total,
            "baseline_criteria": {
                "hook": baseline.score.hook,
                "cliffhanger": baseline.score.cliffhanger,
                "clarity": baseline.score.clarity,
            },
            "episodes": [segment.spec.episode_number for segment in resolved],
            "segments": serialize_segments(resolved, episodes),
            "duration_cs": pacing.duration_cs,
            "duration_seconds": pacing.duration_cs / 100,
            "hook": evidence_record(candidate.hook),
            "cliffhanger": evidence_record(candidate.cliffhanger),
            "hook_summary": response.hook_summary if response else None,
            "cliffhanger_summary": response.cliffhanger_summary if response else None,
            "rationale": candidate.rationale,
            "hook_score": score.hook,
            "cliffhanger_score": score.cliffhanger,
            "clarity_score": score.clarity,
            "drama_score": score.drama,
            "pacing_penalty": score.pacing_penalty,
            "hook_archetype": response.hook_archetype if response else None,
            "opening_tension": response.opening_tension if response else None,
            "ending_tension": response.ending_tension if response else None,
            "join_coherent": response.join_coherent if response else None,
            "judge_uncertainty": response.uncertainty if response else None,
            "tension_delta": score.tension_delta,
            "tension_bonus": score.tension_bonus,
            "spoken_words": pacing.spoken_words,
            "beat_count": pacing.beat_count,
            "non_dialogue_count": pacing.non_dialogue_count,
            "dialogue_wps": pacing.dialogue_wps,
            "action_beat_ratio": pacing.action_beat_ratio,
            "footage_verification_required": "footage_verification_required" in pacing.flags,
            "review_flags": list(candidate.flags),
        })
    return records


def _spreadsheet_safe(value: object) -> object:
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def ranked_csv(records: list[dict]) -> bytes:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS)
    writer.writeheader()
    for record in records:
        row = {name: record.get(name) for name in _CSV_FIELDS}
        row.update({
            "episodes": ", ".join(map(str, record["episodes"])),
            "segments": " | ".join(
                f'E{part["episode"]:03d} {part["start"]}–{part["end"]}' for part in record["segments"]
            ),
            "hook": record["hook"]["text"],
            "hook_beat_id": record["hook"]["beat_id"],
            "cliffhanger": record["cliffhanger"]["text"],
            "cliffhanger_beat_id": record["cliffhanger"]["beat_id"],
            "dialogue_wps": f'{record["dialogue_wps"]:.2f}',
            "review_flags": " | ".join(record["review_flags"]),
        })
        writer.writerow({key: _spreadsheet_safe(value) for key, value in row.items()})
    return handle.getvalue().encode("utf-8-sig")


def json_document(content: object) -> str:
    return json.dumps(content, ensure_ascii=False, indent=2) + "\n"


def ranked_json(analysis: BatchAnalysis) -> str:
    return json_document({"score_source": analysis.ledger["score_source"], "promos": analysis.records})


def _audit(
    inputs: tuple[ParseResult, ...], result: SearchResult, records: list[dict],
    evaluations: dict[str, Evaluation] | None,
) -> str:
    lines = [
        "# Batch review flags",
        "",
        "Review excerpts for an immediate hook, an unresolved ending, and stakes a new viewer can",
        "understand. Confirm footage for cuts marked for that check.",
        "",
        "## Source issues and coverage",
        "",
    ]
    for parsed in inputs:
        for issue in parsed.issues:
            location = f"{parsed.filename}:{issue.line}" if issue.line is not None else parsed.filename
            lines.append(f"- {location} — {issue.severity} {issue.code}. {issue.message}")
    for episode in result.episodes:
        if not episode.selected_ids:
            lines.append(
                f"- Episode {episode.episode_number} — no continuous candidate selected from "
                f"{episode.scanned_windows} timing-valid windows."
            )
    if lines[-1] == "":
        lines.append("- All inputs parsed with continuous coverage and no source issues.")

    lines.extend(("", "## Adjacent stitches and recaps", ""))
    for stitch in result.stitches:
        previous, following = stitch.episodes
        lines.append(
            f"- E{previous:03d} → E{following:03d} — {stitch.reason}. "
            f"Recap {stitch.recap.status} ({stitch.recap.reason}). "
            f"Start {stitch.original_start_beat_id} → {stitch.revised_start_beat_id or 'unavailable'}. "
            f"Selected {stitch.selected_id or 'none'}."
        )
        if stitch.recap.evidence_ids:
            lines.append(f"  Evidence IDs {', '.join(stitch.recap.evidence_ids)}.")
    if not result.stitches:
        lines.append("- No consecutive episode pairs were available.")

    lines.extend(("", "## Model evaluation", ""))
    if evaluations is None:
        lines.append("- Offline baseline run.")
    else:
        validated = sum(evaluation.response is not None for evaluation in evaluations.values())
        used = sum(evaluation.usable for evaluation in evaluations.values())
        lines.append(
            f"- {validated} of {len(evaluations)} selected judgments passed schema and evidence validation. "
            f"{used} were used for ranking."
        )
        for candidate_id, evaluation in evaluations.items():
            if evaluation.error:
                detail = f" ({evaluation.details})" if evaluation.details else ""
                lines.append(f"- {candidate_id} — {evaluation.error}{detail}. Retained baseline score.")

    lines.extend(("", "## Candidate flags", ""))
    for record in records:
        if flags := record["review_flags"]:
            lines.append(f'- Rank {record["rank"]}, {record["candidate_id"]} — {", ".join(flags)}.')
    if not any(record["review_flags"] for record in records):
        lines.append("- No candidate review flags.")
    return "\n".join(lines) + "\n"


def _write_bundle(output: Path, analysis: BatchAnalysis) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".beatledger-", dir=output.parent) as temp:
        staged = Path(temp) / "run"
        staged.mkdir()
        (staged / "ranked_promos.json").write_text(ranked_json(analysis), encoding="utf-8")
        (staged / "ranked_promos.csv").write_bytes(ranked_csv(analysis.records))
        (staged / "run_ledger.json").write_text(
            json_document(analysis.ledger), encoding="utf-8",
        )
        (staged / "failures_and_flags.md").write_text(analysis.audit, encoding="utf-8")
        if output.exists():
            raise FileExistsError(f"Run directory already exists: {output}")
        staged.rename(output)


def _model_ledger(settings: JudgeSettings, evaluations: dict[str, Evaluation], limit: int | None) -> tuple[dict, dict]:
    from beatledger.judge import PROMPT_VERSION

    attempts = [attempt for evaluation in evaluations.values() for attempt in evaluation.attempts]
    input_tokens = sum(attempt.input_tokens or 0 for attempt in attempts)
    output_tokens = sum(attempt.output_tokens or 0 for attempt in attempts)
    unknown = sum(attempt.input_tokens is None or attempt.output_tokens is None for attempt in attempts)
    estimated_cost = None
    if settings.local or not attempts:
        estimated_cost = 0
    elif settings.input_usd_per_million is not None and settings.output_usd_per_million is not None and unknown == 0:
        estimated_cost = round((
            input_tokens * settings.input_usd_per_million
            + output_tokens * settings.output_usd_per_million
        ) / 1_000_000, 8)
    usage = {
        "requests": len(attempts),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "requests_with_unreported_usage": unknown,
        "api_cost_usd": estimated_cost,
        "median_request_seconds": round(median(attempt.latency_seconds for attempt in attempts), 4) if attempts else None,
    }
    judge = {
        "model": settings.model,
        "endpoint_type": "loopback" if settings.local else "remote_https",
        "prompt_version": PROMPT_VERSION,
        "response_format": settings.response_format,
        "temperature": settings.temperature,
        "timeout_seconds": settings.timeout_seconds,
        "max_output_tokens": settings.max_output_tokens,
        "limit": limit,
        "selected": len(evaluations),
        "validated": sum(evaluation.response is not None for evaluation in evaluations.values()),
        "used_for_ranking": sum(evaluation.usable for evaluation in evaluations.values()),
        "evaluations": [
            {
                "candidate_id": candidate_id,
                "validated": evaluation.response is not None,
                "used_for_ranking": evaluation.usable,
                "submitted_scores": (
                    [evaluation.response.hook_score, evaluation.response.cliffhanger_score, evaluation.response.clarity_score]
                    if evaluation.response is not None else None
                ),
                "error": evaluation.error,
                "details": evaluation.details,
                "attempts": [asdict(attempt) for attempt in evaluation.attempts],
            }
            for candidate_id, evaluation in evaluations.items()
        ],
    }
    return usage, judge


def analyze_pack(
    inputs: tuple[ParseResult, ...], *, settings: JudgeSettings | None = None,
    judge_limit: int | None = None, ingest_seconds: float = 0.0,
) -> BatchAnalysis:
    started = time.perf_counter()
    episodes = {entry.episode.number: entry.episode for entry in inputs if entry.episode is not None}
    inventory = build_inventory(inputs)
    result = search_pack(episodes.values())
    searched = time.perf_counter()
    evaluations = None
    if settings is not None:
        from beatledger.judge import judge_shortlist
        from beatledger.judge_provider import ChatCompletionsProvider

        evaluations = judge_shortlist(result.candidates, ChatCompletionsProvider(settings), judge_limit)
    judged = time.perf_counter()
    records = promo_records(result, episodes, evaluations)
    coverage = [
        {**asdict(entry), "not_shortlisted": entry.eligible_windows - len(entry.selected_ids)}
        for entry in result.episodes
    ]
    ledger = {
        "score_source": "judged_and_baseline" if any(row["score_source"] == MODEL_MODE for row in records) else MODE,
        "inputs": inventory,
        "selection_settings": {
            "continuous_limit_per_episode": CONTINUOUS_LIMIT,
            "near_duplicate_iou_percent": DUPLICATE_OVERLAP_PERCENT,
            "stitch_part_cs": [MIN_STITCH_PART_CS, MAX_STITCH_PART_CS],
        },
        "continuous_search": coverage,
        "stitch_search": [asdict(entry) for entry in result.stitches],
        "counts": {
            "continuous": sum(len(candidate.segments) == 1 for candidate in result.candidates),
            "stitched": sum(len(candidate.segments) == 2 for candidate in result.candidates),
            "review_hold": sum(row["priority_group"] == "review_hold" for row in records),
        },
        "model_usage": {"requests": 0, "input_tokens": 0, "output_tokens": 0, "api_cost_usd": 0},
        "latency_seconds": {
            "read_and_parse": round(ingest_seconds, 4),
            "candidate_search": round(searched - started, 4),
            "model_judging": round(judged - searched, 4),
            "read_through_verified_records": round(ingest_seconds + time.perf_counter() - started, 4),
        },
    }
    if settings is not None and evaluations is not None:
        ledger["model_usage"], ledger["judge"] = _model_ledger(settings, evaluations, judge_limit)
    return BatchAnalysis(episodes, records, ledger, _audit(inputs, result, records, evaluations))


def main(argv: list[str] | None = None) -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--input", required=True, type=Path, help="Screenplay .txt file or pack directory")
    cli.add_argument("--output", type=Path, help="New run directory under raw_outputs/ by default")
    cli.add_argument("--judge", action="store_true", help="Evaluate shortlisted cuts using a configured model")
    cli.add_argument("--judge-limit", type=int, help="Limit judging to a mixed sample of the shortlist")
    cli.add_argument("--allow-remote-judge", action="store_true", help="Allow an approved HTTPS model for shareable inputs")
    args = cli.parse_args(argv)
    if args.judge_limit is not None and (args.judge_limit < 1 or not args.judge):
        cli.error("--judge-limit requires --judge and a positive limit.")
    if args.allow_remote_judge and not args.judge:
        cli.error("--allow-remote-judge requires --judge.")

    settings = None
    if args.judge:
        try:
            from beatledger.judge_provider import JudgeSettings

            settings = JudgeSettings.from_environment(allow_remote=args.allow_remote_judge)
        except ValueError as exc:
            cli.error(str(exc))

    output = args.output or Path("raw_outputs") / datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%S%fZ")
    if output.exists():
        cli.error(f"Use a new run directory. This path already exists: {output}")

    started = time.perf_counter()
    try:
        inputs = read_pack(args.input)
    except (OSError, ValueError) as exc:
        cli.error(str(exc))
    try:
        analysis = analyze_pack(
            inputs, settings=settings, judge_limit=args.judge_limit, ingest_seconds=time.perf_counter() - started,
        )
    except ImportError:
        if settings is None:
            raise
        cli.error("Install the judge dependency with python -m pip install -r requirements.txt.")
    try:
        _write_bundle(output, analysis)
    except OSError as exc:
        cli.error(str(exc))
    print(f"Wrote {len(analysis.records)} ranked promos from {len(analysis.episodes)} episodes to {output}")
    judge = analysis.ledger.get("judge")
    if judge is not None:
        print(
            f'Validated {judge["validated"]} of {judge["selected"]} selected model judgments. '
            f'Used {judge["used_for_ranking"]} for ranking.'
        )
    return int(
        analysis.ledger["inputs"]["summary"]["errors"] > 0
        or bool(judge and judge["selected"] and not judge["used_for_ranking"])
    )


if __name__ == "__main__":
    raise SystemExit(main())
