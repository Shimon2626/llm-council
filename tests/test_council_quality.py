"""Council output-quality guards (stdlib unittest; no network).

Reproduces the 2026-10-05 failure: of 8 seats, 3 returned empty text, 3 were cut
off at max_tokens, and all of them were treated as ordinary successes.
Run: .venv/bin/python -m unittest discover -s tests
"""
import asyncio
import unittest
from unittest import mock

from backend import openrouter, council


def reply(content, finish="stop", completion=100):
    return {"content": content, "reasoning_details": None, "finish_reason": finish,
            "usage": {"prompt_tokens": 10, "completion_tokens": completion, "reasoning_tokens": 0}}


class FakeProvider:
    """Plays back scripted replies per (provider, model); records the max_tokens used."""
    def __init__(self, script):
        self.script, self.calls = script, []

    async def __call__(self, provider, model, messages, timeout, max_tokens):
        self.calls.append((provider, model, max_tokens))
        item = self.script[(provider, model)].pop(0)
        if isinstance(item, Exception):
            raise item
        return dict(item)


def run(coro):
    return asyncio.run(coro)


class QueryModelTests(unittest.TestCase):
    def q(self, script, spec, cap=100):
        fake = FakeProvider(script)
        with mock.patch.object(openrouter, "_call_provider", fake):
            return run(openrouter.query_model(spec, [{"role": "user", "content": "x"}], max_tokens=cap)), fake

    def test_empty_with_length_is_retried_with_bigger_cap(self):
        r, fake = self.q({("openrouter", "m"): [reply(None, "length"), reply("full answer")]}, "m")
        self.assertEqual(r["content"], "full answer")
        self.assertFalse(r["empty"]); self.assertFalse(r["truncated"])
        self.assertEqual([c[2] for c in fake.calls], [100, 200])
        self.assertEqual(r["usage"]["completion_tokens"], 200)  # both attempts billed

    def test_truncated_twice_is_flagged_not_passed_off_as_complete(self):
        r, _ = self.q({("openrouter", "m"): [reply("half", "length"), reply("half more", "length")]}, "m")
        self.assertTrue(r["truncated"]); self.assertFalse(r["empty"])

    def test_empty_stop_falls_back_to_other_provider(self):
        r, fake = self.q({("nvidia", "m"): [reply("", "stop")], ("openrouter", "m"): [reply("ok")]},
                         {"nvidia": "m", "openrouter": "m"})
        self.assertEqual(r["provider"], "openrouter"); self.assertEqual(r["content"], "ok")

    def test_always_empty_is_flagged_empty_not_none(self):
        r, _ = self.q({("openrouter", "m"): [reply(None, "stop")]}, "m")
        self.assertTrue(r["empty"])

    def test_hard_errors_everywhere_is_none(self):
        r, _ = self.q({("openrouter", "m"): [RuntimeError("boom")]}, "m")
        self.assertIsNone(r)


def resp(model, content, status="ok"):
    return {"model": model, "response": content, "status": status}


class StageTests(unittest.TestCase):
    def test_stage2_excludes_empty_and_marks_truncated(self):
        seen = {}

        async def fake_parallel(specs, messages, max_tokens=None):
            seen["prompt"] = messages[0]["content"]
            return {"s1": {"content": "A is best.\n\nFINAL RANKING:\n1. Response A\n2. Response B",
                           "provider": "openrouter", "model": "s1", "usage": {}}}

        stage1 = [resp("m/ok", "good"), resp("m/none", "", "empty"),
                  resp("m/cut", "half", "truncated"), resp("m/err", "", "error")]
        with mock.patch.object(council, "query_models_parallel", fake_parallel):
            _, label_to_model = run(council.stage2_collect_rankings("Q", stage1))
        self.assertEqual(label_to_model, {"Response A": "m/ok", "Response B": "m/cut"})
        self.assertIn("cut off at the output limit", seen["prompt"])
        self.assertNotIn("Response C:\n", seen["prompt"])  # only A and B were sent

    def test_stage2_skipped_when_fewer_than_two_usable(self):
        r, m = run(council.stage2_collect_rankings("Q", [resp("a", "x"), resp("b", "", "empty")]))
        self.assertEqual((r, m), ([], {}))

    def test_chairman_is_told_how_many_real_reviews_exist(self):
        seen = {}

        async def fake_query(spec, messages, **kw):
            seen["prompt"] = messages[0]["content"]
            return {"content": "synth", "provider": "openrouter", "model": "c", "usage": {}, "finish_reason": "stop"}

        stage1 = [resp("a", "x"), resp("b", "", "empty"), resp("c", "half", "truncated")]
        with mock.patch.object(council, "query_model", fake_query):
            out = run(council.stage3_synthesize_final("Q", stage1, []))
        p = seen["prompt"]
        self.assertIn("3 seats asked; 1 complete, 1 cut off, 1 returned nothing", p)
        self.assertIn("NOT COVERED", p)
        self.assertIn("No seat had web access", p)
        self.assertEqual(out["status"], "ok")


if __name__ == "__main__":
    unittest.main()
