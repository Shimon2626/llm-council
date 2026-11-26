"""Estimate must price the same caps a run would use. No network, no paid calls."""
import asyncio
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from backend import main, pricing

RATES = {"a/seat1": {"prompt": 1e-6, "completion": 2e-6},
         "b/seat2": {"prompt": 3e-6, "completion": 4e-6},
         "c/chair": {"prompt": 5e-6, "completion": 6e-6}}
SEATS = [{"openrouter": "a/seat1"}, {"openrouter": "b/seat2"}]
CHAIR = {"openrouter": "c/chair"}


def run(c):
    return asyncio.run(c)


def patched():
    return mock.patch.object(pricing, "_get_openrouter_pricing", mock.AsyncMock(return_value=RATES))


class EstimateOverrideTests(unittest.TestCase):
    def test_caps_change_the_total_and_are_reported(self):
        with patched():
            base = run(pricing.estimate_cost("x" * 400, SEATS, CHAIR))
            over = run(pricing.estimate_cost("x" * 400, SEATS, CHAIR, stage1_max_tokens=1000,
                                             stage2_max_tokens=1000, chairman_max_tokens=3000))
        self.assertNotEqual(base["estimated_cost_usd"], over["estimated_cost_usd"])
        self.assertEqual(over["caps"], {"stage1_max_tokens": 1000, "stage2_max_tokens": 1000,
                                        "chairman_max_tokens": 3000})
        # seat1: in 100*1e-6 + out 1000*2e-6
        self.assertAlmostEqual(over["seats"][0]["input_cost_usd"], 100e-6, places=6)
        self.assertAlmostEqual(over["seats"][0]["max_output_cost_usd"], 2e-3, places=6)
        self.assertEqual(over["chairman"]["model"], "c/chair")
        self.assertTrue(over["includes_stage2"])

    def test_endpoint_equals_pricing_with_same_overrides(self):
        ov = {"stage1_max_tokens": 2000, "stage2_max_tokens": 1500, "chairman_max_tokens": 4000,
              "stage1_reasoning_effort": "high", "chairman_reasoning_effort": "low"}
        with patched():
            direct = run(pricing.estimate_cost("hello world " * 50, SEATS, CHAIR, False, **ov))
            r = TestClient(main.app).post("/api/estimate-cost", json={
                "content": "hello world " * 50, "council_models": SEATS, "chairman_model": CHAIR,
                "web_search": "off", **ov})
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["estimated_cost_usd"], direct["estimated_cost_usd"])
        self.assertEqual(body["caps"], direct["caps"])
        self.assertEqual(body["reasoning"], {"stage1": "high", "chairman": "low"})
        self.assertEqual(body["priced_for"], "selected council (2 seats + chairman)")

    def test_default_label(self):
        with patched():
            r = TestClient(main.app).post("/api/estimate-cost", json={"content": "hi", "web_search": "off"})
        self.assertTrue(r.json()["priced_for"].startswith("default council ("))

    def test_empty_council_is_rejected(self):
        with patched(), mock.patch.object(main.config, "COUNCIL_MODELS", []):
            r = TestClient(main.app).post("/api/estimate-cost", json={"content": "hi", "web_search": "off"})
        self.assertEqual(r.status_code, 422)
        self.assertIn("No models selected", r.text)


if __name__ == "__main__":
    unittest.main()
