"""max_tokens is clamped to the catalog's max completion (stdlib unittest; no network).
Run: .venv/bin/python -m unittest discover -s tests
"""
import asyncio
import unittest
from unittest import mock

from backend import openrouter, openrouter_catalog, pricing

NEMO = "nvidia/nemotron-3-ultra-550b-a55b"
GEMMA = "google/gemma-4-31b-it"
KIMI = "moonshotai/kimi-k3"
LIMITS = {NEMO: 16384, GEMMA: 16384, KIMI: 943718}


def reply(finish="stop"):
    return {"content": "ok", "reasoning_details": None, "finish_reason": finish,
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "reasoning_tokens": 0}}


class Recorder:
    def __init__(self, finishes):
        self.finishes, self.caps = list(finishes), []

    async def __call__(self, provider, model, messages, timeout, max_tokens, reasoning_effort="", web_search=False):
        self.caps.append(max_tokens)
        return reply(self.finishes.pop(0))


class CatalogCase(unittest.TestCase):
    def setUp(self):
        saved = dict(openrouter_catalog._max_completion)
        openrouter_catalog._max_completion.clear()
        openrouter_catalog._max_completion.update(LIMITS)
        self.addCleanup(lambda: (openrouter_catalog._max_completion.clear(),
                                 openrouter_catalog._max_completion.update(saved)))


class ClampHelperTests(CatalogCase):
    def test_lowers_to_limit(self):
        self.assertEqual(openrouter_catalog.clamp_max_tokens(40000, NEMO), 16384)

    def test_below_limit_unchanged(self):
        self.assertEqual(openrouter_catalog.clamp_max_tokens(5000, NEMO), 5000)

    def test_unknown_or_none_passes_through(self):
        self.assertEqual(openrouter_catalog.clamp_max_tokens(40000, "x/unknown"), 40000)
        self.assertEqual(openrouter_catalog.clamp_max_tokens(40000, None), 40000)

    def test_falls_back_to_second_id(self):
        self.assertEqual(openrouter_catalog.clamp_max_tokens(40000, "nim-only-slug", GEMMA), 16384)

    def test_record_limits_ignores_missing_values(self):
        openrouter_catalog._max_completion.clear()
        openrouter_catalog.record_limits([
            {"id": "a", "top_provider": {"max_completion_tokens": 100}},
            {"id": "b", "top_provider": {"max_completion_tokens": None}},
            {"id": "c", "top_provider": None},
            {"id": "d"},
        ])
        self.assertEqual(openrouter_catalog._max_completion, {"a": 100})


class QueryModelTests(CatalogCase):
    def _run(self, spec, finishes, max_tokens=40000, ceiling=64000):
        rec = Recorder(finishes)
        with mock.patch.object(openrouter, "_call_provider", rec), \
             mock.patch.object(openrouter_catalog, "ensure_limits", mock.AsyncMock()):
            asyncio.run(openrouter.query_model(spec, [{"role": "user", "content": "q"}],
                                               max_tokens=max_tokens, retry_ceiling=ceiling))
        return rec.caps

    def test_nemotron_sent_16384_not_40000(self):
        self.assertEqual(self._run({"nvidia": NEMO, "openrouter": NEMO}, ["stop"]), [16384])

    def test_gemma_sent_16384(self):
        self.assertEqual(self._run({"nvidia": GEMMA, "openrouter": GEMMA}, ["stop"]), [16384])

    def test_large_limit_model_keeps_requested(self):
        self.assertEqual(self._run({"openrouter": KIMI}, ["stop"]), [40000])

    def test_unknown_model_keeps_requested(self):
        self.assertEqual(self._run({"openrouter": "x/unknown"}, ["stop"]), [40000])

    def test_retry_never_exceeds_limit(self):
        caps = self._run({"openrouter": NEMO}, ["length", "length"])
        self.assertTrue(all(c <= 16384 for c in caps), caps)

    def test_retry_doubles_when_room_remains(self):
        caps = self._run({"openrouter": GEMMA}, ["length", "stop"], max_tokens=3000)
        self.assertEqual(caps, [3000, 6000])

    def test_retry_stops_at_limit_when_doubling_overshoots(self):
        caps = self._run({"openrouter": GEMMA}, ["length", "stop"], max_tokens=10000)
        self.assertEqual(caps, [10000, 16384])


class EstimateTests(CatalogCase):
    RATES = {NEMO: {"prompt": 1e-6, "completion": 2e-6},
             KIMI: {"prompt": 1e-6, "completion": 2e-6}}

    def _est(self, seats, chair, s1=40000):
        with mock.patch.object(pricing, "_get_openrouter_pricing", mock.AsyncMock(return_value=self.RATES)):
            return asyncio.run(pricing.estimate_cost("hi", seats, chair, stage1_max_tokens=s1))

    def test_seat_row_priced_at_clamped_cap(self):
        est = self._est([{"openrouter": NEMO}], {"openrouter": KIMI})
        row = est["seats"][0]
        self.assertEqual(row["max_tokens_used"], 16384)
        self.assertAlmostEqual(row["max_output_cost_usd"], 2e-6 * 16384, places=5)

    def test_unclamped_model_priced_at_requested(self):
        est = self._est([{"openrouter": KIMI}], {"openrouter": KIMI})
        self.assertEqual(est["seats"][0]["max_tokens_used"], 40000)


if __name__ == "__main__":
    unittest.main()
