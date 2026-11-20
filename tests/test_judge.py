import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from email.message import Message
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

from pydantic import ValidationError

from beatledger.inventory import read_pack
from beatledger.judge import HookArchetype, JudgeResponse, evaluate, judge_request, judge_shortlist, pilot_candidates
from beatledger.judge_provider import Attempt, ChatCompletionsProvider, JudgeSettings, ProviderReply
from beatledger.search import search_pack
from run_batch import _model_ledger, main, promo_records


PACK = Path(__file__).parent / "fixtures" / "stitches"


def response_data(candidate):
    return {
        "candidate_id": candidate.id,
        "hook_evidence_id": candidate.hook.id,
        "cliffhanger_evidence_id": candidate.cliffhanger.id,
        "hook_archetype": "Secret Reveal",
        "hook_score": 8,
        "cliffhanger_score": 7,
        "clarity_score": 6,
        "opening_tension": 3,
        "ending_tension": 8,
        "hook_summary": "An accusation raises immediate stakes.",
        "cliffhanger_summary": "The ending leaves a question.",
        "rationale": "An accusation sets up a question that remains unresolved.",
        "join_coherent": True if len(candidate.segments) == 2 else None,
        "uncertainty": None,
    }


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.episodes = {
            item.episode.number: item.episode for item in read_pack(PACK) if item.episode is not None
        }
        cls.search = search_pack(cls.episodes.values())
        cls.continuous = next(candidate for candidate in cls.search.candidates if len(candidate.segments) == 1)
        cls.stitch = next(candidate for candidate in cls.search.candidates if len(candidate.segments) == 2)

    def test_literal_and_integer_validation_are_strict(self):
        from typing import get_args

        data = response_data(self.continuous)
        for archetype in get_args(HookArchetype):
            with self.subTest(archetype=archetype):
                self.assertEqual(JudgeResponse.model_validate({**data, "hook_archetype": archetype}).hook_archetype, archetype)
        for field, value in (
            ("hook_archetype", "secret reveal"), ("hook_archetype", "Drama"),
            ("opening_tension", "3"), ("ending_tension", True), ("ending_tension", 11),
            ("opening_tension", 3.0), ("opening_tension", -1),
        ):
            with self.subTest(field=field, value=value), self.assertRaises(ValidationError):
                JudgeResponse.model_validate_json(json.dumps({**data, field: value}))

    def test_delta_is_derived_read_only_and_serialized(self):
        for start, end, delta in ((3, 8, 5), (8, 3, -5), (4, 4, 0)):
            with self.subTest(start=start, end=end):
                model = JudgeResponse.model_validate({
                    **response_data(self.continuous), "opening_tension": start, "ending_tension": end,
                })
                self.assertEqual(model.tension_delta, delta)
                self.assertEqual(model.model_dump()["tension_delta"], delta)
        self.assertNotIn("tension_delta", JudgeResponse.model_json_schema()["properties"])
        with self.assertRaises(ValidationError):
            JudgeResponse.model_validate({**response_data(self.continuous), "tension_delta": 99})

    def test_request_centiseconds_and_ids_are_relative_to_the_verified_stitch(self):
        messages, schema = judge_request(self.stitch)
        context = json.loads(messages[1]["content"])
        self.assertEqual(context["duration_cs"], self.stitch.pacing.duration_cs)
        self.assertEqual(schema["properties"]["hook_evidence_id"]["enum"], context["eligible_hook_ids"])
        self.assertEqual(schema["properties"]["cliffhanger_evidence_id"]["enum"], context["eligible_cliffhanger_ids"])
        self.assertNotIn("E102-B001", {beat["id"] for beat in context["segments"][1]})
        self.assertTrue(all(
            beat["start_cs_in_cut"] >= self.stitch.segments[0].duration_cs for beat in context["segments"][1]
        ))
        for phrase in ("strictly from the cited hook beat", "strictly\nfrom the cited cliffhanger beat"):
            self.assertIn(phrase, messages[0]["content"])

    def test_valid_judgment_keeps_boundaries_and_holds_incoherent_stitch(self):
        data = {**response_data(self.stitch), "join_coherent": False}
        with patch.object(ChatCompletionsProvider, "complete", return_value=ProviderReply(
            json.dumps(data), (Attempt("response", 0.1, 100, 50),),
        )):
            outcome = evaluate(self.stitch, ChatCompletionsProvider(JudgeSettings()))
        self.assertEqual(outcome.response.tension_delta, 5)
        self.assertEqual(outcome.candidate.score.tension_bonus, 0.25)
        self.assertEqual(outcome.candidate.segments, self.stitch.segments)
        self.assertEqual(outcome.candidate.pacing, self.stitch.pacing)
        self.assertTrue(outcome.candidate.held_for_review)

    def test_invalid_evidence_keeps_baseline_and_consumed_usage(self):
        candidate = self.continuous
        data = {**response_data(candidate), "hook_evidence_id": candidate.cliffhanger.id}
        attempt = Attempt("response", 0.1, 100, 50)
        with patch.object(ChatCompletionsProvider, "complete", return_value=ProviderReply(json.dumps(data), (attempt,))):
            outcome = evaluate(candidate, ChatCompletionsProvider(JudgeSettings()))
        self.assertEqual(outcome.error, "invalid_judge_evidence")
        self.assertIs(outcome.candidate, candidate)
        self.assertEqual(outcome.attempts, (attempt,))

    def test_strong_model_ending_with_modest_stitch_baseline_requires_review(self):
        for cut, held in ((self.stitch, True), (self.continuous, False)):
            candidate = replace(cut, score=replace(cut.score, cliffhanger=6))
            data = {**response_data(candidate), "cliffhanger_score": 8}
            with self.subTest(candidate=candidate.id), patch.object(
                ChatCompletionsProvider, "complete", return_value=ProviderReply(json.dumps(data), ()),
            ):
                outcome = evaluate(candidate, ChatCompletionsProvider(JudgeSettings()))
            self.assertTrue(outcome.usable)
            self.assertEqual(outcome.candidate.score.cliffhanger, 8)
            self.assertEqual("stitch_ending_review" in outcome.candidate.flags, held)
            self.assertEqual(outcome.candidate.held_for_review, held)

    def test_permanent_provider_failure_marks_later_selected_cuts(self):
        pilot = pilot_candidates(self.search.candidates, 3)
        provider = ChatCompletionsProvider(JudgeSettings())
        reply = ProviderReply(None, (Attempt("http_401", .1, http_status=401),), "http_401", True)
        with patch.object(provider, "complete", return_value=reply) as complete:
            evaluations = judge_shortlist(self.search.candidates, provider, 3)
        complete.assert_called_once()
        self.assertEqual(list(evaluations), [cut.id for cut in pilot])
        self.assertEqual(
            [result.error for result in evaluations.values()],
            ["http_401", "skipped_after_provider_failure", "skipped_after_provider_failure"],
        )
        usage, ledger = _model_ledger(JudgeSettings(), evaluations, 3)
        self.assertEqual((usage["requests"], usage["requests_with_unreported_usage"], usage["api_cost_usd"]), (1, 1, 0))
        self.assertEqual(ledger["validated"], 0)

    def test_identical_maximum_scores_leave_the_entire_sample_on_baseline(self):
        by_id = {candidate.id: candidate for candidate in self.search.candidates}

        def saturated_reply(_messages, schema):
            candidate = by_id[schema["properties"]["candidate_id"]["enum"][0]]
            data = {**response_data(candidate), "hook_score": 10, "cliffhanger_score": 10, "clarity_score": 10}
            return ProviderReply(json.dumps(data), (Attempt("response", 0.1, 100, 50),))

        provider = ChatCompletionsProvider(JudgeSettings())
        with patch.object(provider, "complete", side_effect=saturated_reply):
            results = judge_shortlist(self.search.candidates, provider, 4)
        usage, ledger = _model_ledger(JudgeSettings(), results, 4)
        self.assertEqual((ledger["validated"], ledger["used_for_ranking"]), (4, 0))
        self.assertEqual(usage["requests"], 4)
        self.assertTrue(all(
            outcome.error == "saturated_judge_scores" and outcome.candidate is by_id[candidate_id]
            for candidate_id, outcome in results.items()
        ))
        exports = promo_records(self.search, self.episodes, results)
        self.assertTrue(all(record["score"] == record["baseline_score"] for record in exports))
        self.assertTrue(all(record["model_rank"] is None for record in exports))

    def test_cost_uses_reported_tokens_and_leaves_missing_usage_unknown(self):
        settings = JudgeSettings(
            base_url="https://example.com/v1", allow_remote=True, api_key="synthetic-secret",
            input_usd_per_million=2, output_usd_per_million=4,
        )
        candidate = self.continuous
        provider = ChatCompletionsProvider(settings)
        for tokens, cost in (((1_000, 500), 0.004), ((1_000, None), None)):
            with self.subTest(tokens=tokens):
                reply = ProviderReply(json.dumps(response_data(candidate)), (Attempt("response", 0.1, *tokens),))
                with patch.object(provider, "complete", return_value=reply):
                    evaluation = evaluate(candidate, provider)
                usage, ledger = _model_ledger(settings, {candidate.id: evaluation}, 1)
                self.assertEqual(usage["api_cost_usd"], cost)
                self.assertNotIn("synthetic-secret", json.dumps(ledger))

    def test_cli_preserves_baseline_comparisons_and_writes_auditable_model_scores(self):
        def controlled_judge(_self, _messages, schema):
            candidate = next(cut for cut in self.search.candidates if cut.id == schema["properties"]["candidate_id"]["enum"][0])
            return ProviderReply(json.dumps(response_data(candidate)), (Attempt("response", 0.1, 100, 50),))

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "judge"
            with patch.object(ChatCompletionsProvider, "complete", controlled_judge), redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--input", str(PACK), "--output", str(output), "--judge", "--judge-limit", "1"]), 0)
            ledger_json = json.loads((output / "run_ledger.json").read_text(encoding="utf-8"))
            records = json.loads((output / "ranked_promos.json").read_text(encoding="utf-8"))["promos"]
            csv = (output / "ranked_promos.csv").read_text(encoding="utf-8-sig")

        self.assertEqual(len(records), len(self.search.candidates))
        accepted = next(row for row in records if row["judge_status"] == "validated")
        self.assertEqual(accepted["model_rank"], 1)
        self.assertEqual(accepted["hook_archetype"], "Secret Reveal")
        self.assertEqual(accepted["tension_delta"], 5)
        self.assertIn("Secret Reveal", csv)
        self.assertTrue(all(row["baseline_rank"] > 0 for row in records))
        self.assertIn("baseline_not_sampled", {row["judge_status"] for row in records})
        self.assertEqual(ledger_json["judge"]["validated"], 1)
        self.assertEqual(ledger_json["model_usage"]["input_tokens"], 100)


class ProviderTests(unittest.TestCase):
    def test_hosted_endpoint_requires_explicit_https_opt_in(self):
        with self.assertRaisesRegex(ValueError, "explicit"):
            JudgeSettings(base_url="https://example.com/v1")
        with self.assertRaisesRegex(ValueError, "HTTPS"):
            JudgeSettings(base_url="http://example.com/v1", allow_remote=True)
        self.assertFalse(JudgeSettings(base_url="https://example.com/v1", allow_remote=True).local)

    def test_rate_limit_retries_once_honors_bounded_header_and_records_tokens(self):
        provider = ChatCompletionsProvider(JudgeSettings())
        headers = Message()
        headers["Retry-After"] = "10"
        error = HTTPError("http://127.0.0.1", 429, "rate limit", headers, io.BytesIO())
        payload = json.dumps({
            "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 40},
        }).encode()
        with patch.object(provider._opener, "open", side_effect=(error, io.BytesIO(payload))) as send:
            with patch("beatledger.judge_provider.time.sleep") as delay:
                result = provider.complete([], {})
        self.assertEqual(result.content, "{}")
        self.assertEqual([attempt.status for attempt in result.attempts], ["http_429", "response"])
        self.assertEqual((result.attempts[-1].input_tokens, result.attempts[-1].output_tokens), (120, 40))
        delay.assert_called_once_with(5.0)
        self.assertEqual(send.call_count, 2)
        request = send.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:11434/v1/chat/completions")
        self.assertEqual(json.loads(request.data)["response_format"]["type"], "json_schema")

    def test_repeated_timeout_records_two_unknown_usage_attempts(self):
        provider = ChatCompletionsProvider(JudgeSettings(timeout_seconds=1))
        with patch.object(provider._opener, "open", side_effect=TimeoutError):
            with patch("beatledger.judge_provider.time.sleep") as delay:
                reply = provider.complete([], {})
        self.assertEqual(reply.error, "timeout")
        self.assertTrue(reply.stop_run)
        self.assertEqual([attempt.input_tokens for attempt in reply.attempts], [None, None])
        delay.assert_called_once()

    def test_truncation_keeps_usage_but_never_returns_partial_output(self):
        data = b'{"choices":[{"message":{"content":"{}"},"finish_reason":"length"}],"usage":{"prompt_tokens":22,"completion_tokens":55}}'
        self.assertEqual(ChatCompletionsProvider._decode(data), (None, "output_truncated", 22, 55))
