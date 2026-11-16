"""What is saved with a run: typed prompt, attachment lengths and hashes, and
per-seat finish_reason / tokens / duration (mocked providers)."""
import asyncio
import hashlib
import unittest
from unittest import mock

from backend import openrouter, council
from backend.preflight import run_record


class RunRecordTests(unittest.TestCase):
    def test_full_typed_prompt_and_attachment_hashes_are_kept(self):
        typed = "Review the plan. " * 100  # well past any preview length
        content = f"{typed}\n\n**Attached file: Plan.docx**\n```\nplan body\n```"
        rec = run_record(content, {"enabled": False})
        self.assertEqual(rec["typed_prompt"], typed.strip())
        self.assertEqual(rec["attachments"], [{"name": "Plan.docx", "text_chars": 9,
                                               "text_sha256": hashlib.sha256(b"plan body").hexdigest()}])
        self.assertEqual(rec["web_search"], {"enabled": False})

    def test_empty_typed_prompt_is_recorded_as_empty(self):
        self.assertEqual(run_record("\n\n**Attached file: a.txt**\n```\nx\n```")["typed_prompt"], "")


class SeatRowTests(unittest.TestCase):
    def test_row_has_finish_reason_tokens_and_duration(self):
        async def fake(provider, model, messages, timeout, max_tokens, reasoning_effort="", web_search=False):
            return {"content": "hi", "reasoning_details": None, "finish_reason": "stop",
                    "usage": {"prompt_tokens": 7, "completion_tokens": 3, "reasoning_tokens": 2}}

        with mock.patch.object(openrouter, "_call_provider", fake), mock.patch.object(council, "reserve_seats", lambda: []):
            row = asyncio.run(council.stage1_collect_responses("q", [{"openrouter": "m"}]))[0]
        self.assertEqual((row["finish_reason"], row["prompt_tokens"], row["completion_tokens"], row["reasoning_tokens"]),
                         ("stop", 7, 3, 2))
        self.assertIsInstance(row["duration_s"], float)
        self.assertEqual(row["seat"], "m")


if __name__ == "__main__":
    unittest.main()
