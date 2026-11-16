"""Seat failure handling: one retry, provider error body kept, reserve
replacement, duplicate-model removal (stdlib unittest; mocked providers, no network).
Run: .venv/bin/python -m unittest discover -s tests
"""
import asyncio
import unittest
from unittest import mock

import httpx

from backend import openrouter, council, config


def reply(content, finish="stop"):
    return {"content": content, "reasoning_details": None, "finish_reason": finish,
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "reasoning_tokens": 0}}


def http_error(status, body):
    req = httpx.Request("POST", "http://x")
    return httpx.HTTPStatusError("boom", request=req, response=httpx.Response(status, text=body, request=req))


class Fake:
    def __init__(self, script):
        self.script, self.calls = script, []

    async def __call__(self, provider, model, messages, timeout, max_tokens, reasoning_effort="", web_search=False):
        self.calls.append((provider, model))
        item = self.script.get((provider, model), [RuntimeError("no script")]).pop(0)
        if isinstance(item, Exception):
            raise item
        return dict(item)


def run(c):
    return asyncio.run(c)


class RetryAndErrorBodyTests(unittest.TestCase):
    def ask(self, script, spec="m"):
        fake, log = Fake(script), []
        with mock.patch.object(openrouter, "_call_provider", fake), mock.patch.object(openrouter, "RETRY_DELAY_S", 0):
            r = run(openrouter.query_model(spec, [{"role": "user", "content": "x"}], error_log=log))
        return r, fake, log

    def test_transient_error_is_retried_once_then_succeeds(self):
        r, fake, log = self.ask({("openrouter", "m"): [http_error(503, "overloaded"), reply("ok")]})
        self.assertEqual(r["content"], "ok")
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(log[0]["http_status"], 503)

    def test_provider_error_body_is_logged(self):
        r, _, log = self.ask({("openrouter", "m"): [http_error(429, '{"error":"rate limited, try later"}')] * 2})
        self.assertIsNone(r)
        self.assertEqual([e["attempt"] for e in log], [1, 2])
        self.assertIn("rate limited", log[0]["body"])

    def test_non_retryable_4xx_is_not_retried(self):
        _, fake, log = self.ask({("openrouter", "m"): [http_error(410, "gone")]})
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(log[0]["http_status"], 410)


class ReserveTests(unittest.TestCase):
    def stage1(self, script, seats, reserves):
        fake = Fake(script)
        with mock.patch.object(openrouter, "_call_provider", fake), \
             mock.patch.object(openrouter, "RETRY_DELAY_S", 0), \
             mock.patch.object(council, "reserve_seats", lambda: [{"openrouter": r} for r in reserves]):
            return run(council.stage1_collect_responses("q", seats)), fake

    def test_failed_seat_is_replaced_from_reserve_and_reason_shown(self):
        rows, _ = self.stage1(
            {("openrouter", "a"): [reply("A answer")],
             ("openrouter", "b"): [http_error(410, "model retired")],
             ("openrouter", "r1"): [reply("R1 answer")]},
            [{"openrouter": "a"}, {"openrouter": "b"}], ["r1", "r2"])
        ok = [r for r in rows if r["status"] == "ok"]
        self.assertEqual(len(ok), 2)  # council stays at 2 complete seats
        failed = next(r for r in rows if r["seat"] == "b")
        self.assertEqual(failed["status"], "error")
        self.assertIn("model retired", failed["error_detail"])
        self.assertEqual(failed["replaced_by"], "r1")
        self.assertIn("b", next(r for r in rows if r["seat"] == "r1")["replaces"])

    def test_no_reserve_left_means_seat_stays_failed(self):
        rows, _ = self.stage1({("openrouter", "a"): [http_error(410, "x")]}, [{"openrouter": "a"}], [])
        self.assertEqual([r["status"] for r in rows], ["error"])


class DedupeTests(unittest.TestCase):
    def test_free_variant_and_other_provider_are_the_same_base_model(self):
        specs = [{"openrouter": "nvidia/nemotron-3-ultra-550b-a55b"},
                 {"openrouter_free": "nvidia/nemotron-3-ultra-550b-a55b:free", "openrouter": "nvidia/nemotron-3-ultra-550b-a55b"},
                 {"nvidia": "nvidia/nemotron-3-ultra-550b-a55b"},
                 {"openrouter": "moonshotai/kimi-k3"}]
        kept, dropped = config.dedupe_seats(specs)
        self.assertEqual([config.seat_base_model(s) for s in kept], ["nvidia/nemotron-3-ultra-550b-a55b", "moonshotai/kimi-k3"])
        self.assertEqual(len(dropped), 2)

    def test_duplicate_seat_is_not_asked_and_reserve_takes_its_place(self):
        fake = Fake({("openrouter", "m"): [reply("M")], ("openrouter", "r1"): [reply("R")]})
        with mock.patch.object(openrouter, "_call_provider", fake), \
             mock.patch.object(council, "reserve_seats", lambda: [{"openrouter": "r1"}]):
            rows = run(council.stage1_collect_responses("q", [{"openrouter": "m"}, {"openrouter": "m:free"}]))
        self.assertEqual(sorted(c[1] for c in fake.calls), ["m", "r1"])  # the :free twin was never called
        self.assertEqual([r["status"] for r in rows if r["status"] != "duplicate"], ["ok", "ok"])
        self.assertTrue(any(r["status"] == "duplicate" for r in rows))


if __name__ == "__main__":
    unittest.main()
