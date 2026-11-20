"""Streamlit workspace for inspecting verified promo cuts and recording decisions."""

from __future__ import annotations

import math
import os
import time
from html import escape
from pathlib import Path

import streamlit as st

from beatledger.judge_provider import JudgeSettings
from beatledger.parser import format_clock, parse_pack
from beatledger.verifier import CutError, SegmentSpec, Timeline
from beatledger.workbench import (
    SAVED_SAMPLE, UNJUDGED, ReviewCut, decision_export, filter_records, load_saved_sample,
    original_cut, review_decision, sample_sources, trim_cut,
)
from run_batch import BatchAnalysis, analyze_pack, json_document, ranked_csv, ranked_json


ROOT = Path(__file__).resolve().parent
MODEL_SAMPLE_LIMIT = 12
HOSTED_SAMPLE_LIMIT = 4
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


def _run_sources(sources: list[tuple[str, bytes]], source: str, model: bool, saved: bool) -> None:
    settings = None
    if model:
        try:
            settings = JudgeSettings.from_environment()
        except ValueError as exc:
            st.warning(f"Local judge configuration is unavailable. {exc} Running the baseline.")
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
        _run_sources(sample_sources(ROOT), "Bundled sample", False, False)
    else:
        _install_run(_hosted_sample(settings), "Bundled sample", False)


def _source_controls() -> None:
    hosted = os.getenv("BEATLEDGER_DEPLOYMENT") == "hosted"
    with st.sidebar:
        st.header("Screenplays")
        choice = st.radio("Input source", ("Bundled sample", "Upload files"), key="input_source")
        uploads = []
        if choice == "Upload files":
            uploads = st.file_uploader("Episode screenplays", type=["txt"], accept_multiple_files=True, key="uploads")
            if hosted:
                st.caption("Upload only scripts you can share with this deployment. Uploads use the offline baseline.")
        else:
            st.caption("Three synthetic episodes with a recap and a silent action sequence.")
        judge = False
        if not hosted:
            judge = st.checkbox(
                "Use local model judge",
                help="Requires a local model server. Judges up to 12 candidate cuts in this run.",
                key="enable_judge",
            )
        if st.session_state.get("decisions") or st.session_state.get("revisions"):
            st.caption("Export your review decisions before starting another run.")
        run = st.button(
            "Run analysis", type="primary", width="stretch",
            disabled=choice == "Upload files" and not uploads, key="run_analysis",
        )
        saved = choice == "Bundled sample" and (ROOT / SAVED_SAMPLE).is_file() and st.button(
            "Open saved model run", width="stretch", key="saved_sample",
        )
        hosted_judge = hosted and choice == "Bundled sample" and st.button(
            "Judge sample live", width="stretch", key="hosted_sample",
        )
        if "analysis" not in st.session_state or run or saved or hosted_judge:
            work = "Loading saved judgments…" if saved else "Finding and verifying promo cuts…"
            with st.spinner(work):
                try:
                    if hosted_judge:
                        _run_hosted_sample()
                    elif not run or choice == "Bundled sample":
                        _run_sources(sample_sources(ROOT), "Bundled sample", run and judge, saved)
                    else:
                        _run_sources([(item.name, item.getvalue()) for item in uploads], "Uploaded screenplays", judge, False)
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


def _cut_details(cut: ReviewCut, record: dict) -> None:
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


def _adjust_cut(analysis: BatchAnalysis, record: dict, cut: ReviewCut, serial: int) -> None:
    candidate_id = record["candidate_id"]
    timelines = st.session_state.timelines
    version = f"{serial}:{candidate_id}:{cut.version}"
    with st.expander("Adjust cut"):
        with st.form(f"trim:{version}"):
            specs = []
            for index, segment in enumerate(cut.segments, 1):
                number = segment.spec.episode_number
                source = timelines[number]
                start_times, end_times = sorted(source.starts), sorted(source.ends)
                left, right = st.columns(2)
                start = left.selectbox(
                    f"Segment {index} start", start_times, index=start_times.index(segment.start_cs),
                    format_func=format_clock, key=f"start:{version}:{index}",
                )
                end = right.selectbox(
                    f"Segment {index} end", end_times, index=end_times.index(segment.end_cs),
                    format_func=format_clock, key=f"end:{version}:{index}",
                )
                specs.append(SegmentSpec(number, source.starts[start], source.ends[end]))
            apply = st.form_submit_button("Verify and apply trim")
            restore = st.form_submit_button("Restore original cut") if cut.revised else False

        if not (apply or restore):
            return
        try:
            if restore:
                updated = original_cut(record, timelines)
            else:
                resume = next(
                    (entry["recap"]["resume_cs"] for entry in analysis.ledger["stitch_search"]
                     if list(entry["episodes"]) == record["episodes"] and entry["recap"]["status"] == "confirmed"),
                    None,
                )
                updated = trim_cut(record, tuple(specs), timelines, resume)
        except CutError as exc:
            st.error(str(exc))
            return
        if updated.version == cut.version:
            st.info("These are the current cut boundaries.")
            return
        if updated.revised:
            st.session_state.revisions[candidate_id] = updated
        else:
            st.session_state.revisions.pop(candidate_id, None)
        st.session_state.decisions.pop(candidate_id, None)
        st.session_state.feedback = "Verified cut saved for a new review."
        st.rerun()


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
    with left:
        _cut_details(cut, record)
    with right:
        _save_review(record, cut, serial)
        _adjust_cut(analysis, record, cut, serial)


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


def main() -> None:
    st.set_page_config(page_title="BeatLedger Editor Workbench", layout="wide", initial_sidebar_state="auto")
    _render_title()
    _source_controls()
    if "analysis" not in st.session_state:
        return
    analysis = st.session_state.analysis
    _run_header(analysis)
    review, audit = st.tabs(("Review cuts", "Run audit"))
    with review:
        _review_tab(analysis)
    with audit:
        _audit_tab(analysis)


if __name__ == "__main__":
    main()
