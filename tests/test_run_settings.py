"""Per-run overrides: parsing, clamping, env fallbacks, retry ceiling, stage 3 truncation.
No network, no paid calls (providers mocked).
Run: .venv/bin/python -m unittest discover -s tests
"""
import asyncio
import unittest
from unittest import mock

from backend import openrouter as o, council, run_settings as rs


def run(c):
    return asyncio.run(c)


class ParseTests(unittest.TestCase):
    def test_none_and_digit_strings(self):
        self.assertIsNone(rs.parse_int(None, "f"))
        self.assertEqual(rs.parse_int(" 40000 ", "f"), 40000)
        self.assertEqual(rs.parse_int(5000, "f"), 5000)

    def test_rejects_non_integers(self):
        for bad in ("12.5", "abc", 1.5, True, "-3", ""):
            with self.assertRaises(ValueError, msg=repr(bad)):
                rs.parse_int(bad, "stage1_max_tokens")

    def test_chairman_reasoning(self):
        self.assertEqual(rs.resolve_settings()["chairman_reasoning_effort"], o.CHAIRMAN_REASONING_EFFORT)
        self.assertEqual(rs.resolve_settings(chairman_reasoning_effort="default")["chairman_reasoning_effort"], "")
        self.assertEqual(rs.resolve_settings(chairman_reasoning_effort="HIGH")["chairman_reasoning_effort"], "high")
        with self.assertRaises(ValueError) as cm:
            rs.resolve_settings(chairman_reasoning_effort="extreme")
        self.assertIn("chairman_reasoning_effort", str(cm.exception))

    def test_reasoning(self):
        self.assertIsNone(rs.parse_reasoning(None))
        self.assertEqual(rs.parse_reasoning(" LOW "), "low")
        with self.assertRaises(ValueError):
            rs.parse_reasoning("extreme")


class ResolveTests(unittest.TestCase):
    def test_absent_uses_env_defaults(self):
        r = rs.resolve_settings()
        self.assertEqual(r["stage1_max_tokens"], o.DEFAULT_MAX_TOKENS)
        self.assertEqual(r["retry_max_tokens_ceiling"], o.RETRY_MAX_TOKENS_CEILING)
        self.assertEqual(r["stage2_max_tokens"], o.STAGE2_MAX_TOKENS)
        self.assertEqual(r["chairman_max_tokens"], o.CHAIRMAN_MAX_TOKENS)
        self.assertEqual(r["stage1_reasoning_effort"], o.STAGE1_REASONING_EFFORT)
        self.assertEqual(r["clamped"], [])

    def test_overrides_applied(self):
        r = rs.resolve_settings(stage1_max_tokens=40000, stage2_max_tokens=12000, chairman_max_tokens=40000,
                                retry_max_tokens_ceiling=64000, stage1_reasoning_effort="high",
                                max_office_extract_chars=400000)
        self.assertEqual((r["stage1_max_tokens"], r["stage2_max_tokens"], r["chairman_max_tokens"]), (40000, 12000, 40000))
        self.assertEqual(r["retry_max_tokens_ceiling"], 64000)
        self.assertEqual(r["stage1_reasoning_effort"], "high")
        self.assertEqual(r["max_office_extract_chars"], 400000)

    def test_clamped_to_hard_limits(self):
        r = rs.resolve_settings(stage1_max_tokens=10 ** 7, chairman_max_tokens=5, max_office_extract_chars=10 ** 9)
        self.assertEqual(r["stage1_max_tokens"], rs.HARD_MAX_TOKENS)
        self.assertEqual(r["chairman_max_tokens"], rs.MIN_VALUE)
        self.assertEqual(r["max_office_extract_chars"], rs.HARD_MAX_CHARS)
        self.assertEqual(len(r["clamped"]), 3)

    def test_default_reasoning_means_no_parameter(self):
        self.assertEqual(rs.resolve_settings(stage1_reasoning_effort="default")["stage1_reasoning_effort"], "")

    def test_stage2_precedence(self):
        self.assertEqual(rs.resolve_settings(stage2_max_tokens=7000, legacy_council_max_tokens=9000)["stage2_max_tokens"], 7000)
        self.assertEqual(rs.resolve_settings(legacy_council_max_tokens=9000)["stage2_max_tokens"], 9000)


class RetryCeilingTests(unittest.TestCase):
    def ask(self, replies, **kw):
        caps = []

        async def fake(provider, model, messages, timeout, max_tokens, reasoning_effort="", web_search=False):
            caps.append((max_tokens, reasoning_effort))
            return dict(replies.pop(0))

        def rep(text, fin):
            return {"content": text, "reasoning_details": None, "finish_reason": fin,
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "reasoning_tokens": 0}}

        replies[:] = [rep(*x) for x in replies]
        with mock.patch.object(o, "_call_provider", fake), mock.patch.object(o, "RETRY_DELAY_S", 0):
            r = run(o.query_model("m", [{"role": "user", "content": "x"}], **kw))
        return r, caps

    def test_retry_doubles_but_never_exceeds_ceiling(self):
        r, caps = self.ask([("cut", "length"), ("done", "stop")], max_tokens=40000, retry_ceiling=64000)
        self.assertEqual([c[0] for c in caps], [40000, 64000])
        self.assertEqual(r["content"], "done")

    def test_no_retry_when_cap_already_at_ceiling(self):
        r, caps = self.ask([("cut", "length")], max_tokens=64000, retry_ceiling=64000)
        self.assertEqual(len(caps), 1)
        self.assertTrue(r["truncated"])

    def test_reasoning_effort_passed_through(self):
        _, caps = self.ask([("ok", "stop")], max_tokens=5000, reasoning_effort="low")
        self.assertEqual(caps[0][1], "low")


class Stage3Tests(unittest.TestCase):
    def test_truncated_chairman_is_flagged_and_uses_override_cap(self):
        seen = {}

        async def fake_query(spec, messages, **kw):
            seen.update(kw)
            return {"provider": "p", "model": "m", "content": "partial", "finish_reason": "length",
                    "truncated": True, "max_tokens_used": kw.get("max_tokens"),
                    "usage": {"completion_tokens": 40000, "reasoning_tokens": 30000}}

        s1 = [{"model": "a/b", "response": "x", "status": "ok"}]
        with mock.patch.object(council, "query_model", fake_query):
            r = run(council.stage3_synthesize_final("q", s1, [], chairman_max_tokens=40000, retry_ceiling=64000))
        self.assertEqual(r["status"], "truncated")
        self.assertEqual(r["finish_reason"], "length")
        self.assertEqual(seen["max_tokens"], 40000)
        self.assertEqual(seen["retry_ceiling"], 64000)
        self.assertEqual(r["reasoning_tokens"], 30000)


if __name__ == "__main__":
    unittest.main()
