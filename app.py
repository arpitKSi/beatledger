"""Streamlit workspace for inspecting verified promo cuts and recording decisions."""

from __future__ import annotations

import math
import os
import re
import time
from html import escape
from pathlib import Path
from statistics import median

import streamlit as st

from beatledger.judge_provider import JudgeSettings
from beatledger.parser import Episode, format_clock, parse_pack
from beatledger.ranking import BaselineScorer
from beatledger.verifier import CutError, Segment, SegmentSpec, Timeline, cut_beat_kind, verify_cut
from beatledger.workbench import (
    SAVED_SAMPLE, UNJUDGED, ReviewCut, decision_export, filter_records, load_saved_sample,
    original_cut, review_decision, sample_sources, trim_cut,
)
from run_batch import BatchAnalysis, analyze_pack, json_document, ranked_csv, ranked_json


ROOT = Path(__file__).resolve().parent
MODEL_SAMPLE_LIMIT = 12
HOSTED_SAMPLE_LIMIT = 4
OFFLINE_JUDGE = "Offline baseline"
HOSTED_PROVIDERS = {
    "Groq": ("https://api.groq.com/openai/v1", "openai/gpt-oss-20b"),
    "Google Gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "gemini-3.5-flash-lite"),
    "OpenAI": ("https://api.openai.com/v1", "gpt-4.1-mini"),
    "OpenRouter": ("https://openrouter.ai/api/v1", "openrouter/free"),
}
REVIEW_STATUS = {"pending": "Pending", "approved": "Approved", "rejected": "Rejected"}
FLAGS = {
    "rushed_dialogue": "Dialogue may feel rushed",
    "low_criterion_score": "A criterion scored below 6",
    "pacing_timing_review": "Check the timing of the hook or ending",
    "context_dependent_opening": "Opening may depend on earlier context",
    "stitch_continuity_review": "Check continuity across episodes",
    "recap_review_required": "Possible recap at the join",
    "incoherent_stitch": "Judge flagged the join",
    "stitch_ending_review": "Baseline and judge disagree on the stitched ending",
    "judge_uncertainty": "Judge expressed uncertainty",
    "missing_trim_evidence": "Revised cut needs opening or ending evidence",
}


def _render_title() -> None:
    st.html("""
        <style>
            .block-container { padding-top: 3rem; }
            .bl-header { background: #12363b; color: #fff; padding: 1.05rem 1.45rem;
              border-radius: 1rem; margin-bottom: .4rem; border-left: .35rem solid #51c5ae; }
            .bl-header h1 { color: #fff; margin: 0; font-size: clamp(2rem, 4vw, 3rem); padding: 0;
              letter-spacing: -.035em; }
            .bl-header p { margin: .25rem 0 0; color: #d8efea; font-size: 1rem; }
            .bl-evidence { line-height: 1.55; font-size: .96rem; margin: .15rem 0 .7rem;
              white-space: pre-wrap; overflow-wrap: anywhere; }
            .bl-excerpt { max-height: 29rem; overflow-y: auto; padding-right: .6rem; }
            .bl-excerpt h4 { padding: 0; margin: 0 0 .25rem; font-size: 1rem; }
            .bl-excerpt h5 { padding: 0; margin: .6rem 0 .3rem; font-size: .91rem; }
            .bl-excerpt section + section { border-top: 2px solid #b9cfcb; padding-top: .85rem; }
            .bl-excerpt article { border-top: 1px solid #d6e3e1; padding: .5rem 0; }
            .bl-excerpt .bl-meta { color: #465f62; font-size: .82rem; margin: 0 0 .25rem;
              overflow-wrap: anywhere; }
            .bl-excerpt .bl-evidence { margin: 0; }
            @media (max-width: 640px) {
                .st-key-run-metrics [data-testid="stHorizontalBlock"] { flex-wrap: wrap; gap: .65rem; }
                .st-key-run-metrics [data-testid="stColumn"] { flex: 1 1 45%; min-width: 0; }
            }
        </style>
        <header class="bl-header">
            <h1>BeatLedger</h1>
            <p>Find, verify, and review 20–60 second promo cuts.</p>
        </header>
    """)


def _install_run(analysis: BatchAnalysis, source: str, saved: bool) -> None:
    st.session_state.analysis = analysis
    st.session_state.source_label = source
    st.session_state.saved = saved
    st.session_state.timelines = {number: Timeline(episode) for number, episode in analysis.episodes.items()}
    st.session_state.run_number = st.session_state.get("run_number", 0) + 1
    st.session_state.revisions = {}
    st.session_state.decisions = {}
    st.session_state.pop("feedback", None)


def _run_sources(
    sources: list[tuple[str, bytes]], source: str, *, settings: JudgeSettings | None = None, saved: bool = False,
) -> None:
    started = time.perf_counter()
    parsed = parse_pack(sources)
    run = analyze_pack(
        parsed, settings=settings, judge_limit=MODEL_SAMPLE_LIMIT,
        ingest_seconds=time.perf_counter() - started,
    )
    if saved:
        run = load_saved_sample(ROOT, run)
    _install_run(run, source, saved)


@st.cache_data(ttl=900, max_entries=1, show_spinner=False)
def _hosted_sample(settings: JudgeSettings) -> BatchAnalysis:
    # Remote requests always start from the on-disk synthetic fixtures.
    started = time.perf_counter()
    parsed = parse_pack(sample_sources(ROOT))
    return analyze_pack(
        parsed, settings=settings, judge_limit=HOSTED_SAMPLE_LIMIT,
        ingest_seconds=time.perf_counter() - started,
    )


def _run_hosted_sample() -> None:
    try:
        settings = JudgeSettings.from_environment(allow_remote=True)
        if settings.local or not settings.api_key.strip():
            raise ValueError("Hosted judging requires a remote HTTPS model and key.")
    except ValueError:
        st.warning("Live sample judging is unavailable. Showing the baseline. You can also open the saved model run.")
        _run_sources(sample_sources(ROOT), "Bundled sample")
    else:
        _install_run(_hosted_sample(settings), "Bundled sample", False)


def _save_session_key(provider: str, field: str) -> None:
    api_key = st.session_state.get(field, "").strip()
    st.session_state[field] = ""
    if not api_key or not all(33 <= ord(char) <= 126 for char in api_key):
        st.session_state.judge_key_error = provider
        return
    st.session_state.judge_api_keys[provider] = api_key
    st.session_state.pop("judge_key_error", None)


def _hosted_judge_controls() -> tuple[str, JudgeSettings | None]:
    provider = st.selectbox("AI judge", (OFFLINE_JUDGE, *HOSTED_PROVIDERS), key="judge_provider")
    if provider == OFFLINE_JUDGE:
        return provider, None

    base_url, default_model = HOSTED_PROVIDERS[provider]
    model = st.text_input("Model ID", default_model, max_chars=160, key=f"judge_model_{provider}").strip()
    api_keys = st.session_state.setdefault("judge_api_keys", {})
    field = f"judge_key_{provider}"
    with st.form(f"judge_credentials_{provider}", border=False, enter_to_submit=False):
        st.text_input("API key", type="password", key=field, max_chars=1024)
        st.form_submit_button("Save key for this session", on_click=_save_session_key, args=(provider, field))
    if st.session_state.get("judge_key_error") == provider:
        st.warning("Enter a key with no spaces or line breaks.")
    if provider in api_keys and st.button("Remove saved key", key=f"remove_key_{provider}"):
        del api_keys[provider]
    api_key = api_keys.get(provider)
    if not api_key:
        st.caption("Save an API key to enable AI analysis.")
        return provider, None

    st.caption(f"{provider} key saved for this session. AI analysis sends excerpts from up to "
               f"{MODEL_SAMPLE_LIMIT} cuts to {provider} and may use your API credits.")
    if model == api_key or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]*", model):
        st.warning("Enter a valid model ID for this provider.")
        return provider, None
    return provider, JudgeSettings(
        base_url=base_url, model=model, api_key=api_key, response_format="json", temperature=None,
        timeout_seconds=25, max_output_tokens=2048, allow_remote=True,
        use_max_completion_tokens=provider in {"OpenAI", "Groq"},
    )


def _source_controls() -> None:
    hosted = os.getenv("BEATLEDGER_DEPLOYMENT") == "hosted"
    with st.sidebar:
        st.header("Screenplays")
        choice = st.radio("Input source", ("Bundled sample", "Upload files"), key="input_source")
        uploads = []
        if choice == "Upload files":
            uploads = st.file_uploader("Episode screenplays", type=["txt"], accept_multiple_files=True, key="uploads")
            if hosted:
                st.caption("Upload only scripts you have permission to share.")
        else:
            st.caption("Three synthetic episodes with a recap and a silent action sequence.")
        provider, settings = OFFLINE_JUDGE, None
        local_judge = False
        if hosted:
            provider, settings = _hosted_judge_controls()
        else:
            local_judge = st.checkbox(
                "Use local model judge",
                help="Requires a local model server. Judges up to 12 candidate cuts in this run.",
                key="enable_judge",
            )
        if st.session_state.get("decisions") or st.session_state.get("revisions"):
            st.caption("Export your review decisions before starting another run.")
        run = st.button(
            "Run AI analysis" if provider != OFFLINE_JUDGE else "Run analysis", type="primary", width="stretch",
            disabled=(choice == "Upload files" and not uploads) or (provider != OFFLINE_JUDGE and settings is None),
            key="run_analysis",
        )
        saved = choice == "Bundled sample" and (ROOT / SAVED_SAMPLE).is_file() and st.button(
            "Open saved model run", width="stretch", key="saved_sample",
        )
        hosted_judge = hosted and provider == OFFLINE_JUDGE and choice == "Bundled sample" and st.button(
            "Judge sample live", width="stretch", key="hosted_sample",
        )
        if "analysis" not in st.session_state or run or saved or hosted_judge:
            work = "Loading saved judgments…" if saved else "Finding and verifying promo cuts…"
            with st.spinner(work):
                try:
                    if hosted_judge:
                        _run_hosted_sample()
                    else:
                        if run and local_judge:
                            try:
                                settings = JudgeSettings.from_environment()
                            except ValueError as exc:
                                st.warning(f"Local judge configuration is unavailable. {exc} Running the baseline.")
                        if run and choice == "Upload files":
                            sources = [(item.name, item.getvalue()) for item in uploads]
                            _run_sources(sources, "Uploaded screenplays", settings=settings)
                        else:
                            _run_sources(sample_sources(ROOT), "Bundled sample", settings=settings if run else None, saved=saved)
                except (OSError, KeyError, ValueError) as exc:
                    st.error(f"Could not load this run. {exc}")


def _run_header(analysis: BatchAnalysis) -> None:
    judge = analysis.ledger.get("judge", {})
    mode = "Saved model run" if st.session_state.saved else "Live run"
    used = judge.get("used_for_ranking", 0)
    if used:
        label = judge["model"]
    else:
        label = "Baseline fallback" if judge.get("selected") else "Offline baseline"
    st.caption(f"{st.session_state.source_label} · {mode} · {label}")
    approvals = sum(decision.status == "approved" for decision in st.session_state.decisions.values())
    columns = st.container(key="run-metrics").columns(4)
    for column, (label, value) in zip(columns, (
        ("Episodes parsed", len(analysis.episodes)), ("Candidate cuts", len(analysis.records)),
        ("Judged cuts", used), ("Approved", approvals),
    )):
        column.metric(label, value, border=True)
    if judge.get("selected", 0) and not judge.get("used_for_ranking", 0):
        errors = {item["error"] for item in judge.get("evaluations", [])}
        if errors & {"http_401", "http_403"}:
            st.warning("The provider rejected access. Check the API key and model permissions. Showing the baseline.")
        elif errors & {"http_402", "http_429"}:
            st.warning("The provider reported a billing or rate limit. Check your API account. Showing the baseline.")
        elif errors & {"http_400", "http_404", "http_422"}:
            st.warning("The provider rejected the request. Check that the model ID supports chat and JSON mode. Showing the baseline.")
        else:
            st.warning("The model returned no usable judgments. Review the baseline or open the saved sample.")


def _tension_display(record: dict) -> str:
    delta = record["tension_delta"]
    if delta is None:
        return "—"
    return f'{delta:+d} ({record["opening_tension"]}→{record["ending_tension"]})'


def _table_records(records: list[dict]) -> list[dict]:
    decisions, revisions = st.session_state.decisions, st.session_state.revisions
    return [
        {
            "Rank": record["rank"],
            "Episodes": " → ".join(map(str, record["episodes"])),
            "Duration": record["duration_seconds"],
            "Score": record["score"],
            "Archetype": record["hook_archetype"] or UNJUDGED,
            "Tension change": _tension_display(record),
            "WPS": record["dialogue_wps"],
            "Review": (
                REVIEW_STATUS[decisions[record["candidate_id"]].status]
                if record["candidate_id"] in decisions else
                "Edited" if record["candidate_id"] in revisions else
                "Hold" if record["priority_group"] == "review_hold" else "Pending"
            ),
        }
        for record in records
    ]


def _filtered(analysis: BatchAnalysis, serial: int) -> list[dict]:
    c1, c2, c3, c4 = st.columns((1.4, 1, 1, 1))
    archetypes = sorted({record["hook_archetype"] for record in analysis.records if record["hook_archetype"]})
    if any(record["hook_archetype"] is None for record in analysis.records):
        archetypes.append(UNJUDGED)
    kinds = c1.multiselect("Hook archetype", archetypes, placeholder="All archetypes", key=f"archetype-{serial}")
    tension = c2.slider(
        "Tension delta", -10, 10, (-10, 10), key=f"tension-{serial}",
        disabled=not any(record["tension_delta"] is not None for record in analysis.records),
    )
    max_wps = max(1.0, math.ceil(max(record["dialogue_wps"] for record in analysis.records) * 10) / 10)
    pace = c3.slider("Dialogue WPS", 0.0, max_wps, (0.0, max_wps), step=.01, key=f"wps-{serial}")
    status = c4.selectbox(
        "Review status", ("all", *REVIEW_STATUS), format_func=lambda value: REVIEW_STATUS.get(value, "All cuts"),
        key=f"status-{serial}",
    )
    return filter_records(
        analysis.records, st.session_state.decisions, archetypes=kinds, tension=tension, pacing=pace, status=status,
    )


def _esc_paragraph(value: str) -> None:
    st.html(f'<p class="bl-evidence">{escape(value)}</p>')


def _evidence_panel(
    title: str, evidence: dict | None, summary: str | None,
) -> None:
    st.markdown(f"**{title}**")
    if evidence is None:
        st.caption("No eligible evidence at this boundary.")
        return
    st.caption(f'{evidence["beat_id"]} · {evidence["start"]}–{evidence["end"]} · line {evidence["source_line"]}')
    _esc_paragraph(evidence["text"])
    if summary:
        st.caption("Judge summary")
        _esc_paragraph(summary)


def _cut_excerpt(segments: tuple[Segment, ...], episodes: dict[int, Episode], *, previewing: bool) -> None:
    with st.expander("Full cut excerpt", expanded=True):
        duration = sum(part.duration_cs for part in segments) / 100
        label = "Trim preview" if previewing else "Saved cut"
        st.caption(f"{label} · {duration:.2f} s · All screenplay beats overlapping these timecodes")
        sections = []
        for part in segments:
            episode = episodes[part.spec.episode_number]
            scenes = {scene.number: scene.heading for scene in episode.scenes}
            rows = [
                f"<h4>Episode {episode.number} · {format_clock(part.start_cs)}–{format_clock(part.end_cs)}</h4>",
                f'<p class="bl-meta">{escape(episode.filename)}</p>',
            ]
            current_scene = None
            for beat in sorted(part.beats, key=lambda item: (item.start_cs, item.source_ordinal)):
                if beat.scene_number is not None and beat.scene_number != current_scene:
                    current_scene = beat.scene_number
                    heading = scenes.get(current_scene)
                    detail = f" · {escape(heading)}" if heading else ""
                    rows.append(f"<h5>Scene {current_scene:02d}{detail}</h5>")
                original = f"{format_clock(beat.start_cs)}–{format_clock(beat.end_cs)}"
                inside_start, inside_end = max(part.start_cs, beat.start_cs), min(part.end_cs, beat.end_cs)
                overlap = ""
                if (inside_start, inside_end) != (beat.start_cs, beat.end_cs):
                    overlap = (f'<p class="bl-meta">Overlaps cut edge · Inside the cut '
                               f'{format_clock(inside_start)}–{format_clock(inside_end)}</p>')
                kind = cut_beat_kind(beat).capitalize()
                category = f' <span class="bl-meta">· {kind}</span>' if beat.label.casefold() != kind.casefold() else ""
                rows.append(
                    f'<article><strong>{escape(beat.label)}</strong>{category}'
                    f'<p class="bl-meta">{original} · {beat.id} · line {beat.source_line}</p>'
                    f'{overlap}<p class="bl-evidence">{escape(beat.text)}</p></article>'
                )
            sections.append(f"<section>{''.join(rows)}</section>")
        st.html(f'<div class="bl-excerpt" role="region" aria-label="Screenplay beats" tabindex="0">'
                f'{"".join(sections)}</div>')


def _cut_details(cut: ReviewCut, record: dict, episodes: dict[int, Episode], preview: tuple[Segment, ...]) -> None:
    if cut.revised:
        st.caption("Revised cut · pacing and evidence refreshed")
    else:
        basis = "Model judged" if record["judge_status"] == "validated" else "Deterministic baseline"
        st.caption(
            f'{basis} · {record["score"]:.2f} overall · Hook {record["hook_score"]} · '
            f'Cliffhanger {record["cliffhanger_score"]} · Clarity {record["clarity_score"]}'
        )
        if record["judge_status"] == "validated":
            st.caption(f'{record["hook_archetype"]} · Tension {_tension_display(record)}')
            st.caption(
                f'Baseline rank #{record["baseline_rank"]} · Baseline score {record["baseline_score"]:.2f}'
            )
    st.caption(
        f"{cut.pacing.duration_cs / 100:.2f} s · {cut.pacing.dialogue_wps:.2f} dialogue WPS · "
        f"{cut.pacing.action_beat_ratio:.0%} non-dialogue beats"
    )
    for segment in cut.segments:
        st.text(
            f"Episode {segment.spec.episode_number}  "
            f"{format_clock(segment.start_cs)} → {format_clock(segment.end_cs)}  "
            f"({segment.duration_cs / 100:.2f} s)"
        )
    _cut_excerpt(preview or cut.segments, episodes, previewing=bool(preview))
    with st.expander("Cut metrics"):
        _cut_metrics(cut, record)
    with st.expander("Source evidence", expanded=True):
        _evidence_panel("Opening hook", cut.hook, None if cut.revised else record["hook_summary"])
        _evidence_panel("Ending beat", cut.cliffhanger, None if cut.revised else record["cliffhanger_summary"])
        if not cut.revised:
            st.caption("Ranking rationale")
            _esc_paragraph(record["rationale"])
    shown_flags = [FLAGS[flag] for flag in cut.flags if flag in FLAGS]
    if shown_flags:
        st.caption("Review flags")
        st.markdown("\n".join(f"- {flag}" for flag in shown_flags))


def _cut_metrics(cut: ReviewCut, record: dict) -> None:
    revised_score = None
    if cut.revised:
        candidate = BaselineScorer(beat for part in cut.segments for beat in part.beats).score(cut.segments)
        revised_score = candidate.score if candidate else None
        st.caption("The edited range has fresh baseline metrics. Reranking and model judgments apply to the original cut.")
    baseline = record["baseline_criteria"]
    baseline_total = 0.35 * baseline["hook"] + 0.30 * baseline["cliffhanger"] + 0.25 * baseline["clarity"] + 0.10 * record["drama_score"]
    original = (
        ("Overall score", f'{record["baseline_score"]:.2f}', f'{record["score"]:.2f}',
         f"{revised_score.total:.2f}" if revised_score else "—"),
        ("Opening hook · 35%", baseline["hook"], record["hook_score"], revised_score.hook if revised_score else "—"),
        ("Cliffhanger · 30%", baseline["cliffhanger"], record["cliffhanger_score"], revised_score.cliffhanger if revised_score else "—"),
        ("Standalone clarity · 25%", baseline["clarity"], record["clarity_score"], revised_score.clarity if revised_score else "—"),
        ("Drama · 10%", record["drama_score"], record["drama_score"], revised_score.drama if revised_score else "—"),
        ("Pacing deduction", f'{baseline_total - record["baseline_score"]:.2f}', f'{record["pacing_penalty"]:.2f}',
         f"{revised_score.pacing_penalty:.2f}" if revised_score else "—"),
        ("Tension change (opening → ending)", "—", _tension_display(record), "—"),
        ("Tension bonus", "—", f'{record["tension_bonus"]:+.2f}', "—"),
        ("Rank", f'#{record["baseline_rank"]}', f'#{record["rank"]}', "—"),
    )
    rows = []
    for metric, rule_based, ranked, edited in original:
        row = {"Metric": metric, "Original baseline": str(rule_based), "Ranked cut": str(ranked)}
        if cut.revised:
            row["Edited baseline"] = str(edited)
        rows.append(row)
    st.dataframe(rows, hide_index=True, width="stretch", height=min(360, 42 + 36 * len(rows)))
    pacing = cut.pacing
    st.caption(f"Active range contains {pacing.spoken_words} spoken words, {pacing.beat_count} beats, "
               f"and {pacing.non_dialogue_count} non-dialogue beats.")
    if not cut.revised and record["join_coherent"] is not None:
        st.caption("Model assessment of stitch continuity: " + ("coherent" if record["join_coherent"] else "review needed"))


def _save_review(record: dict, cut: ReviewCut, serial: int) -> None:
    candidate_id = record["candidate_id"]
    existing = st.session_state.decisions.get(candidate_id)
    version = f"{serial}:{candidate_id}:{cut.version}"
    with st.form(f"decision:{version}"):
        st.markdown("**Editor decision**")
        status = st.radio(
            "Decision", tuple(REVIEW_STATUS), index=list(REVIEW_STATUS).index(existing.status) if existing else 0,
            format_func=REVIEW_STATUS.get, horizontal=True, key=f"decision-status:{version}",
        )
        note = st.text_area("Review note", value=existing.note if existing else "", key=f"note:{version}")
        confirmed = False
        if cut.footage_required:
            st.warning("This cut has zero dialogue. Check it against the footage before approving.")
            confirmed = st.checkbox(
                "I checked this cut against the footage",
                value=bool(existing and existing.footage_verified), key=f"footage:{version}",
            )
        if st.form_submit_button("Save decision", type="primary"):
            try:
                st.session_state.decisions[candidate_id] = review_decision(cut, status, note, confirmed)
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.session_state.feedback = f"{REVIEW_STATUS[status]} decision saved."
                st.rerun()


def _trim_slider(cut: ReviewCut, timelines: dict[int, Timeline], version: str) -> tuple[SegmentSpec, ...]:
    first, last = cut.segments[0], cut.segments[-1]
    first_episode, last_episode = first.spec.episode_number, last.spec.episode_number
    stitched = len(cut.segments) == 2
    move_join = stitched and st.radio(
        "Stitch adjustment", ("Cut edges", "Stitch join"), horizontal=True, key=f"trim-mode:{version}",
    ) == "Stitch join"
    if move_join:
        left = {time: beat for time, beat in timelines[first_episode].ends.items() if time > first.start_cs}
        right = {time: beat for time, beat in timelines[last_episode].starts.items() if time < last.end_cs}
        left[first.end_cs], right[last.start_cs] = first.spec.end_beat_id, last.spec.start_beat_id
        current_start, current_end = first.end_cs, last.start_cs
    else:
        left = {time: beat for time, beat in timelines[first_episode].starts.items() if not stitched or time < first.end_cs}
        right = {time: beat for time, beat in timelines[last_episode].ends.items() if not stitched or time > last.start_cs}
        left[first.start_cs], right[last.end_cs] = first.spec.start_beat_id, last.spec.end_beat_id
        current_start, current_end = first.start_cs, last.end_cs

    # Give each episode its own half of a stitched slider so the handles can share one control.
    last_left = max(left)
    offset = last_left + 1 - min(right) if stitched else 0
    start, end = st.select_slider(
        "Cut range" if not move_join else "Join range",
        options=sorted(left.keys() | {time + offset for time in right}),
        value=(current_start, current_end + offset), key=f"trim-range:{version}:{move_join}",
        format_func=lambda time: format_clock(time - offset if stitched and time > last_left else time),
        help="Move both handles between screenplay beats. The preview shows the exact source timecodes.",
    )
    start_id = left[min(left, key=lambda time: (abs(time - start), time))]
    end_id = right[min(right, key=lambda time: (abs(time - (end - offset)), time))]
    if move_join:
        return (SegmentSpec(first_episode, first.spec.start_beat_id, start_id),
                SegmentSpec(last_episode, end_id, last.spec.end_beat_id))
    if stitched:
        return (SegmentSpec(first_episode, start_id, first.spec.end_beat_id),
                SegmentSpec(last_episode, last.spec.start_beat_id, end_id))
    return (SegmentSpec(first_episode, start_id, end_id),)


def _adjust_cut(analysis: BatchAnalysis, record: dict, cut: ReviewCut, serial: int) -> tuple[Segment, ...]:
    candidate_id = record["candidate_id"]
    timelines = st.session_state.timelines
    version = f"{serial}:{candidate_id}:{cut.version}"
    with st.expander("Adjust cut"):
        specs = _trim_slider(cut, timelines, version)
        try:
            preview = verify_cut(timelines, specs)
        except CutError as exc:
            preview = ()
            st.warning(str(exc))
        else:
            duration = sum(part.duration_cs for part in preview)
            ranges = " · ".join(f"Ep {part.spec.episode_number} {format_clock(part.start_cs)}–{format_clock(part.end_cs)}"
                                for part in preview)
            st.caption(f"Preview · {duration / 100:.2f} s · {ranges}")
        apply = st.button("Verify and apply trim", disabled=not preview, key=f"apply-trim:{version}")
        restore = cut.revised and st.button("Restore original cut", key=f"restore-trim:{version}")
        excerpt_preview = preview if specs != tuple(part.spec for part in cut.segments) else ()

        if not (apply or restore):
            return excerpt_preview
        try:
            if restore:
                updated = original_cut(record, timelines)
            else:
                resume = next(
                    (entry["recap"]["resume_cs"] for entry in analysis.ledger["stitch_search"]
                     if list(entry["episodes"]) == record["episodes"] and entry["recap"]["status"] == "confirmed"),
                    None,
                )
                updated = trim_cut(record, specs, timelines, resume)
        except CutError as exc:
            st.error(str(exc))
            return excerpt_preview
        if updated.version == cut.version:
            st.info("These are the current cut boundaries.")
            return ()
        if updated.revised:
            st.session_state.revisions[candidate_id] = updated
        else:
            st.session_state.revisions.pop(candidate_id, None)
        st.session_state.decisions.pop(candidate_id, None)
        st.session_state.feedback = "Verified cut saved for a new review."
        st.rerun()
        return ()


def _downloads(analysis: BatchAnalysis) -> None:
    with st.expander("Download run and reviews"):
        a, b, c = st.columns(3)
        a.download_button("Ranked CSV", ranked_csv(analysis.records), "ranked_promos.csv", "text/csv", on_click="ignore")
        b.download_button("Ranked JSON", ranked_json(analysis), "ranked_promos.json", "application/json", on_click="ignore")
        reviews = decision_export(
            analysis.records, analysis.episodes, st.session_state.timelines,
            st.session_state.revisions, st.session_state.decisions,
        )
        c.download_button(
            "Review decisions JSON", json_document(reviews), "review_decisions.json", "application/json",
            disabled=not reviews["reviews"], on_click="ignore",
        )


def _review_tab(analysis: BatchAnalysis) -> None:
    st.subheader("Ranked cuts")
    _downloads(analysis)
    if feedback := st.session_state.pop("feedback", None):
        st.success(feedback)
    if not analysis.records:
        st.info("No verified 20–60 second cuts. Open Run audit to inspect the inputs.")
        return
    serial = st.session_state.run_number
    visible = _filtered(analysis, serial)
    if not visible:
        st.info("No cuts match these filters.")
        return
    st.dataframe(
        _table_records(visible), hide_index=True, width="stretch", height=min(290, 42 + 36 * len(visible)),
        column_config={
            "Rank": st.column_config.NumberColumn(width=62),
            "Score": st.column_config.NumberColumn(format="%.2f"),
            "Duration": st.column_config.NumberColumn(format="%.2f s"),
            "Archetype": st.column_config.TextColumn(width=180),
            "WPS": st.column_config.NumberColumn("Dialogue WPS", format="%.2f"),
        },
    )
    st.caption(f"{len(visible)} of {len(analysis.records)} cuts")
    records = {record["candidate_id"]: record for record in visible}

    def cut_label(candidate_id: str) -> str:
        record = records[candidate_id]
        episodes = " → ".join(map(str, record["episodes"]))
        return f'#{record["rank"]} · Episode {episodes} · {record["duration_seconds"]:.2f} s'

    key = f"inspect-{serial}"
    if st.session_state.get(key) not in records:
        st.session_state[key] = next(iter(records))
    chosen = st.selectbox("Inspect cut", tuple(records), format_func=cut_label, key=key)
    record = records[chosen]
    cut = st.session_state.revisions.get(chosen) or original_cut(record, st.session_state.timelines)
    left, right = st.columns((1.1, 1), gap="large")
    with right:
        preview = _adjust_cut(analysis, record, cut, serial)
        if preview:
            st.caption("The editor decision applies to the saved range until you apply the trim.")
        _save_review(record, cut, serial)
    with left:
        _cut_details(cut, record, analysis.episodes, preview)


def _audit_tab(analysis: BatchAnalysis) -> None:
    ledger = analysis.ledger
    st.subheader("Run audit")
    a, b = st.columns(2)
    a.download_button("Run ledger JSON", json_document(ledger), "run_ledger.json", "application/json", on_click="ignore")
    b.download_button("Review flags Markdown", analysis.audit, "failures_and_flags.md", "text/markdown", on_click="ignore")
    files = ledger["inputs"]["files"]
    with st.expander("Input coverage", expanded=True):
        st.dataframe([
            {
                "File": entry["filename"], "Episode": entry["episode_number"], "Status": entry["status"],
                "Beats": entry["beats"], "Issues": len(entry["issues"]),
            }
            for entry in files
        ], hide_index=True, width="stretch", height=min(310, 42 + len(files) * 36))
        for entry in files:
            for issue in entry["issues"]:
                st.text(f'{entry["filename"]} · line {issue["line"] or "—"} · {issue["message"]}')
    with st.expander("Stitch and recap checks"):
        entries = ledger["stitch_search"]
        if entries:
            st.dataframe([
                {
                    "Episode pair": " → ".join(map(str, entry["episodes"])),
                    "Recap": entry["recap"]["status"], "Decision": entry["reason"],
                    "Original start": entry["original_start_beat_id"], "Reviewed start": entry["revised_start_beat_id"],
                }
                for entry in entries
            ], hide_index=True, width="stretch")
        else:
            st.caption("This run has no consecutive episode pairs.")
    with st.expander("Model usage"):
        usage = ledger["model_usage"]
        st.text(f'{usage["requests"]} requests · {usage["input_tokens"]:,} input tokens · {usage["output_tokens"]:,} output tokens')
        for item in ledger.get("judge", {}).get("evaluations", []):
            if item["error"]:
                st.text(f'{item["candidate_id"]} · {item["error"]}')


def _metric_table(rows: tuple[tuple[str, object], ...]) -> None:
    st.dataframe(
        [{"Metric": name, "Value": str(value)} for name, value in rows],
        hide_index=True, width="stretch", height=38 + 35 * len(rows),
    )


def _metrics_tab(analysis: BatchAnalysis) -> None:
    ledger, records = analysis.ledger, analysis.records
    inputs, counts = ledger["inputs"]["summary"], ledger["counts"]
    judge, usage, timing = ledger.get("judge", {}), ledger["model_usage"], ledger["latency_seconds"]
    judged = [record for record in records if record["judge_status"] == "validated"]
    decisions = [item.status for item in st.session_state.decisions.values()]
    seconds = lambda value: f"{value:,.3f} s" if value is not None else "—"
    cost = usage["api_cost_usd"]

    st.subheader("Run metrics")
    st.caption("Measured on the active run. The bundled sample and uploaded screenplays have separate metrics.")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Coverage and safety")
        _metric_table((
            ("Files parsed / supplied", f'{inputs["parsed_files"]} / {inputs["files"]}'),
            ("Episodes with a verified cut", len({episode for record in records for episode in record["episodes"]})),
            ("Scenes / screenplay beats", f'{inputs["scenes"]:,} / {inputs["beats"]:,}'),
            ("Input warnings / errors", f'{inputs["warnings"]} / {inputs["errors"]}'),
            ("Continuous / stitched cuts", f'{counts["continuous"]} / {counts["stitched"]}'),
            ("Eligible continuous windows", f'{sum(row["eligible_windows"] for row in ledger["continuous_search"]):,}'),
            ("Episode pairs checked for recaps", len(ledger["stitch_search"])),
            ("Source ranges / evidence citations", f'{sum(len(record["segments"]) for record in records)} / {2 * len(records)}'),
            ("Duration range", f'{min(row["duration_seconds"] for row in records):.2f}–'
             f'{max(row["duration_seconds"] for row in records):.2f} s' if records else "—"),
            ("Median dialogue words per second", f'{median(record["dialogue_wps"] for record in records):.2f}' if records else "—"),
            ("Original cuts on review hold", counts["review_hold"]),
            ("Cuts requiring footage confirmation", sum(record["footage_verification_required"] for record in records)),
            ("Session edits / approvals / rejections", f'{len(st.session_state.revisions)} / '
             f'{decisions.count("approved")} / {decisions.count("rejected")}'),
        ))
    with right:
        st.markdown("#### Model usage and timing")
        _metric_table((
            ("Model", judge.get("model", "Offline baseline")),
            ("Selected / validated / used", f'{judge.get("selected", 0)} / {judge.get("validated", 0)} / {len(judged)}'),
            ("Requests / missing token usage", f'{usage["requests"]} / {usage.get("requests_with_unreported_usage", 0)}'),
            ("Input / output tokens", f'{usage["input_tokens"]:,} / {usage["output_tokens"]:,}'),
            ("API cost", f"${cost:,.8g}" if cost is not None else "Unknown"),
            ("Median model request", seconds(usage.get("median_request_seconds"))),
            ("Read and parse", seconds(timing.get("read_and_parse"))),
            ("Candidate search", seconds(timing.get("candidate_search"))),
            ("Model judging", seconds(timing.get("model_judging"))),
            ("Total through verified records", seconds(timing.get("read_through_verified_records"))),
        ))
    st.markdown("#### Baseline versus model")
    if judged:
        _metric_table((
            ("Judged cuts compared", len(judged)),
            ("Higher / unchanged / lower scores", f'{sum(row["score"] > row["baseline_score"] for row in judged)} / '
             f'{sum(row["score"] == row["baseline_score"] for row in judged)} / '
             f'{sum(row["score"] < row["baseline_score"] for row in judged)}'),
            ("Moved up / unchanged / moved down in rank", f'{sum(row["rank"] < row["baseline_rank"] for row in judged)} / '
             f'{sum(row["rank"] == row["baseline_rank"] for row in judged)} / '
             f'{sum(row["rank"] > row["baseline_rank"] for row in judged)}'),
            ("Model judgments on stitch-ending hold", sum("stitch_ending_review" in row["review_flags"] for row in judged)),
        ))
    else:
        st.caption("Open the saved model run or run AI analysis to see how model judging changes scores and ranks.")
    st.markdown("#### How the workflow improves the shortlist")
    st.markdown(
        "- **Verify before ranking.** Screenplay beat IDs anchor every edge and citation. "
        "The verifier enforces dialogue-safe boundaries and the 20–60 second duration limit.\n"
        "- **Reduce duplicate openings.** The search removes near-duplicate cuts and checks stitched episodes for recaps.\n"
        "- **Score the story against source evidence.** Ranking weights hook 35%, cliffhanger 30%, clarity 25%, and drama 10%. "
        "Pacing can deduct points, and rising model-rated tension can add up to 0.35 points.\n"
        "- **Put uncertain cuts in front of an editor.** Stitch disagreements trigger a review hold. "
        "Silent cuts require footage confirmation, and an applied trim resets the previous decision."
    )


def main() -> None:
    st.set_page_config(page_title="BeatLedger Editor Workbench", layout="wide", initial_sidebar_state="auto")
    _render_title()
    _source_controls()
    if "analysis" not in st.session_state:
        return
    analysis = st.session_state.analysis
    _run_header(analysis)
    review, metrics, audit = st.tabs(("Review cuts", "Metrics", "Run audit"))
    with review:
        _review_tab(analysis)
    with metrics:
        _metrics_tab(analysis)
    with audit:
        _audit_tab(analysis)


if __name__ == "__main__":
    main()
