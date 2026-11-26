"""The chairman is told about seat errors, non-compliance, self-votes and a
program-counted verdict tally, and cannot claim unanimity the tally does not show."""
import asyncio
import unittest
from unittest import mock

from backend import council
from backend.compliance import guard_unanimity


def setUpModule():
    # No network in tests: skip the model-list fetch that fills the max-completion limits.
    from unittest import mock as _mock
    from backend import openrouter_catalog as _cat
    global _limits_patch
    _limits_patch = _mock.patch.object(_cat, "ensure_limits", _mock.AsyncMock())
    _limits_patch.start()


def tearDownModule():
    _limits_patch.stop()


QUERY = "Verdict: AGREE, PARTLY AGREE or DISAGREE. Table Q1 to Q12."


def row(model, text, status="ok", **kw):
    return {"model": model, "seat": model, "response": text, "status": status, **kw}


STAGE1 = [
    row("m/a", "AGREE\nfine"),
    row("m/b", "DISAGREE\nevent semantics are not established"),
    row("m/c", "Note first\nAGREE", compliance={"ok": False, "issues": ["line 1 is not a verdict"], "reasked": True}),
    row("m/d", "", status="error", error_detail="HTTP 410 model retired", replaced_by="r1"),
]
STAGE2 = [{"model": "m/a", "self_label": "Response A", "parsed_ranking": ["Response A", "Response B"], "ranking": "x"}]


class NotesTests(unittest.TestCase):
    def test_notes_list_errors_noncompliance_selfvote_and_tally(self):
        text, tally, n_ok = council.chairman_notes(QUERY, STAGE1, STAGE2)
        self.assertIn("SEAT FAILURE: m/d", text); self.assertIn("HTTP 410", text); self.assertIn("Replaced by r1", text)
        self.assertIn("NON-COMPLIANT: m/c", text)
        self.assertIn("SELF-VOTES: 1", text)
        self.assertEqual((tally["AGREE"], tally["DISAGREE"], tally["unparsed"], n_ok), (1, 1, 1, 3))
        self.assertIn("VERDICT TALLY", text)

    def test_no_tally_for_a_question_without_verdicts(self):
        _, tally, _ = council.chairman_notes("Summarise.", STAGE1, [])
        self.assertIsNone(tally)


class ChairmanPromptTests(unittest.TestCase):
    def run_chairman(self, said):
        seen = {}

        async def fake(spec, messages, **kw):
            seen["prompt"] = messages[-1]["content"]
            return {"content": said, "provider": "openrouter", "model": "k", "usage": {}, "finish_reason": "stop",
                    "empty": False, "truncated": False}

        with mock.patch.object(council, "query_model", fake):
            out = asyncio.run(council.stage3_synthesize_final(QUERY, STAGE1, STAGE2))
        return out, seen["prompt"]

    def test_prompt_carries_the_facts_and_the_rule(self):
        _, prompt = self.run_chairman("ok")
        for needle in ("SEAT FAILURE", "NON-COMPLIANT", "SELF-VOTES", "VERDICT TALLY", 'Never write that "all"'):
            self.assertIn(needle, prompt)

    def test_false_unanimity_claim_gets_a_correction(self):
        out, _ = self.run_chairman("The diagnosis was agreed by all eight reviews.")
        self.assertIn("[Council note:", out["response"]); self.assertIn("AGREE 1", out["response"])

    def test_honest_counts_are_left_alone(self):
        out, _ = self.run_chairman("One seat agreed, one disagreed, one gave no verdict.")
        self.assertNotIn("[Council note:", out["response"])


class GuardTests(unittest.TestCase):
    def test_true_unanimity_passes(self):
        self.assertEqual(guard_unanimity("All seats agree.", {"AGREE": 3, "DISAGREE": 0}, 3), "All seats agree.")

    def test_claim_without_a_tally_is_flagged_as_unverified(self):
        self.assertIn("not counted", guard_unanimity("Every reviewer agrees.", None, 3))


if __name__ == "__main__":
    unittest.main()
