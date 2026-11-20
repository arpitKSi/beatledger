import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from beatledger.judge_provider import ChatCompletionsProvider, ProviderReply


APP = Path(__file__).resolve().parents[1] / "app.py"


def button(app, label):
    return next(item for item in app.button if item.label == label)


class WorkbenchAppTests(unittest.TestCase):
    def setUp(self):
        self.app = AppTest.from_file(str(APP), default_timeout=25).run()
        self.assertFalse(self.app.exception)

    def test_sample_runs_without_server_and_saved_model_view_filters_ranked_cuts(self):
        ui = self.app
        self.assertEqual([metric.value for metric in ui.metric], ["3", "9", "0", "0"])
        ui.button(key="saved_sample").click().run()
        self.assertFalse(ui.exception)
        self.assertEqual(ui.metric[2].value, "9")
        self.assertIn("+3 (5→8)", set(ui.dataframe[0].value["Tension change"]))
        self.assertTrue(any(item.value.startswith("Baseline rank #") for item in ui.caption))

        ui.multiselect(key="archetype-2").set_value(["Secret Reveal"])
        ui.slider(key="tension-2").set_value((1, 10))
        ui.slider(key="wps-2").set_value((.60, .80))
        ui.run()
        self.assertFalse(ui.exception)
        visible = ui.dataframe[0].value
        self.assertEqual(len(visible), 1)
        self.assertEqual(visible.iloc[0]["Archetype"], "Secret Reveal")
        self.assertEqual(visible.iloc[0]["WPS"], .70)

    def test_silent_approval_is_blocked_then_trim_resets_it(self):
        ui = self.app
        candidate_id = "E103:0-2050"
        ui.selectbox(key="inspect-1").set_value(candidate_id).run()
        next(item for item in ui.radio if item.label == "Decision").set_value("approved")
        button(ui, "Save decision").click().run()
        self.assertFalse(ui.exception)
        self.assertTrue(any("Confirm that you checked the footage" in error.value for error in ui.error))
        self.assertNotIn(candidate_id, ui.session_state.decisions)

        next(item for item in ui.checkbox if "checked this cut against the footage" in item.label).check()
        button(ui, "Save decision").click().run()
        self.assertFalse(ui.exception)
        self.assertEqual(ui.session_state.decisions[candidate_id].status, "approved")
        self.assertEqual(ui.metric[3].value, "1")

        next(item for item in ui.selectbox if item.label == "Segment 1 end").set_value(180)
        button(ui, "Verify and apply trim").click().run()
        self.assertTrue(any("20 to 60 seconds" in error.value for error in ui.error))
        self.assertEqual(ui.session_state.decisions[candidate_id].status, "approved")

        next(item for item in ui.selectbox if item.label == "Segment 1 end").set_value(2420)
        button(ui, "Verify and apply trim").click().run()
        self.assertFalse(ui.exception)
        self.assertNotIn(candidate_id, ui.session_state.decisions)
        self.assertEqual(ui.session_state.revisions[candidate_id].pacing.duration_cs, 2420)
        self.assertFalse(next(
            item for item in ui.checkbox if "checked this cut against the footage" in item.label
        ).value)
        self.assertEqual(ui.metric[3].value, "0")

        next(item for item in ui.radio if item.label == "Decision").set_value("rejected")
        button(ui, "Save decision").click().run()
        self.assertFalse(ui.exception)
        self.assertEqual(ui.session_state.decisions[candidate_id].status, "rejected")

    def test_hosted_judge_falls_back_and_is_hidden_when_uploading(self):
        with patch.dict(os.environ, {
            "BEATLEDGER_DEPLOYMENT": "hosted",
            "BEATLEDGER_BASE_URL": "https://example.com/v1",
            "BEATLEDGER_API_KEY": "",
        }):
            ui = AppTest.from_file(str(APP), default_timeout=25).run()
            self.assertNotIn("enable_judge", {item.key for item in ui.checkbox})
            ui.button(key="hosted_sample").click().run()
            self.assertFalse(ui.exception)
            self.assertTrue(any("Live sample judging is unavailable" in item.value for item in ui.warning))
            self.assertEqual(ui.metric[2].value, "0")
            ui.radio(key="input_source").set_value("Upload files").run()
            self.assertNotIn("hosted_sample", {item.key for item in ui.button})

    def test_hosted_judge_uses_only_synthetic_candidates_and_caches_the_result(self):
        sent = []

        def complete(messages, schema):
            request = json.loads(messages[1]["content"])
            sent.append(request["candidate_id"])
            return ProviderReply(json.dumps({
                "candidate_id": request["candidate_id"],
                "hook_evidence_id": request["eligible_hook_ids"][0],
                "cliffhanger_evidence_id": request["eligible_cliffhanger_ids"][0],
                "hook_summary": "The opening introduces a dispute.",
                "cliffhanger_summary": "The dispute remains unresolved.",
                "hook_archetype": "Other",
                "rationale": "The dispute raises a question for the next scene.",
                "hook_score": 6, "cliffhanger_score": 6, "clarity_score": 7,
                "opening_tension": 4, "ending_tension": 6,
                "join_coherent": True if len(request["segments"]) == 2 else None,
                "uncertainty": None,
            }), ())

        with patch.dict(os.environ, {
            "BEATLEDGER_DEPLOYMENT": "hosted", "BEATLEDGER_BASE_URL": "https://example.com/v1",
            "BEATLEDGER_API_KEY": "test-secret", "BEATLEDGER_MODEL": "fixture-judge",
        }), patch.object(ChatCompletionsProvider, "complete", side_effect=complete):
            ui = AppTest.from_file(str(APP), default_timeout=25).run()
            ui.button(key="hosted_sample").click().run()
            self.assertFalse(ui.exception)
            self.assertEqual(ui.metric[2].value, "4")
            self.assertTrue(all(part.startswith(("E101:", "E102:", "E103:")) for cut in sent for part in cut.split("+")))
            self.assertEqual(len(sent), 4)
            ui.button(key="hosted_sample").click().run()
            self.assertFalse(ui.exception)
            self.assertEqual(ui.metric[2].value, "4")
            self.assertEqual(len(sent), 4)
