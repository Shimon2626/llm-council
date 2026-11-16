"""Format post-check and one re-ask for the missing part (mocked providers)."""
import asyncio
import unittest
from unittest import mock

from backend import openrouter, council
from backend.compliance import (check_response, format_contract, parse_verdict,
                                section_word_counts, table_rows_present)

QUERY = "Give a verdict: AGREE, PARTLY AGREE or DISAGREE. Then a table for Q1 to Q12."
CONTRACT = format_contract(QUERY)
TABLE = "\n".join(f"| Q{i} | KEEP | ok |" for i in range(1, 13))
GOOD = f"PARTLY AGREE\n\n### 1. Diagnosis (≤120 words)\nShort answer here.\n\n### 2. Table\n{TABLE}\n"


class CheckTests(unittest.TestCase):
    def test_compliant_answer_passes(self):
        self.assertTrue(check_response(GOOD, CONTRACT)["ok"])

    def test_reversed_verdict_agree_partly_fails(self):
        self.assertIsNone(parse_verdict("AGREE PARTLY\n..."))
        self.assertIn("verdict", check_response("AGREE PARTLY\n" + TABLE, CONTRACT)["missing"])

    def test_note_before_verdict_fails(self):
        r = check_response("Note: I could not see the audit file.\nAGREE\n" + TABLE, CONTRACT)
        self.assertIn("verdict", r["missing"])

    def test_markdown_wrapped_verdict_is_accepted(self):
        self.assertEqual(parse_verdict("**DISAGREE**\nx"), "DISAGREE")
        self.assertEqual(parse_verdict("# PARTLY AGREE"), "PARTLY AGREE")

    def test_missing_table_rows_are_named(self):
        r = check_response("AGREE\n| Q1 | KEEP | x |\n| Q2 | CUT | y |\n", CONTRACT)
        self.assertIn("table", r["missing"])
        self.assertTrue(any("Q3" in i for i in r["issues"]))

    def test_heading_mention_of_q1_q12_is_not_a_table(self):
        self.assertEqual(table_rows_present("### 4. Q1-Q12 - KEEP / CHANGE / CUT\ntext"), [])

    def test_answer_over_120_words_is_flagged(self):
        long = "### 1. Diagnosis\n" + "word " * 130 + "\n### 2. Next\nshort\n"
        counts = section_word_counts(long)
        self.assertEqual(counts[1], 130)
        r = check_response("AGREE\n" + long + TABLE, CONTRACT)
        self.assertEqual(list(r["over_cap"]), [1]); self.assertFalse(r["ok"])

    def test_query_without_the_format_has_no_contract(self):
        self.assertIsNone(format_contract("Summarise this plan."))


def reply(content):
    return {"content": content, "reasoning_details": None, "finish_reason": "stop",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "reasoning_tokens": 0}}


class ReaskTests(unittest.TestCase):
    def run_stage1(self, script, query=QUERY):
        calls = []

        async def fake(provider, model, messages, timeout, max_tokens, reasoning_effort="", web_search=False):
            calls.append(messages)
            return dict(script.pop(0))

        with mock.patch.object(openrouter, "_call_provider", fake), mock.patch.object(council, "reserve_seats", lambda: []):
            rows = asyncio.run(council.stage1_collect_responses(query, [{"openrouter": "m"}]))
        return rows, calls

    def test_missing_verdict_and_table_are_reasked_once_for_those_parts_only(self):
        rows, calls = self.run_stage1([reply("### 1. Diagnosis\nplan is fine"), reply("PARTLY AGREE\n" + TABLE)])
        self.assertEqual(len(calls), 2)  # one answer + exactly one re-ask
        self.assertIn("Send ONLY that missing part", calls[1][-1]["content"])
        self.assertTrue(rows[0]["response"].startswith("PARTLY AGREE"))
        self.assertTrue(rows[0]["compliance"]["reasked"])
        self.assertTrue(rows[0]["compliance"]["ok"], rows[0]["compliance"])

    def test_still_noncompliant_after_one_reask_is_marked_not_looped(self):
        rows, calls = self.run_stage1([reply("no verdict"), reply("still nothing useful")])
        self.assertEqual(len(calls), 2)
        self.assertFalse(rows[0]["compliance"]["ok"])

    def test_compliant_answer_is_not_reasked(self):
        rows, calls = self.run_stage1([reply(GOOD)])
        self.assertEqual(len(calls), 1); self.assertTrue(rows[0]["compliance"]["ok"])

    def test_ordinary_question_is_never_checked(self):
        rows, calls = self.run_stage1([reply("just an answer")], query="What is 2+2?")
        self.assertNotIn("compliance", rows[0]); self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
