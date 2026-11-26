import asyncio
from backend import openrouter


def setUpModule():
    # No network in tests: skip the model-list fetch that fills the max-completion limits.
    from unittest import mock as _mock
    from backend import openrouter_catalog as _cat
    global _limits_patch
    _limits_patch = _mock.patch.object(_cat, "ensure_limits", _mock.AsyncMock())
    _limits_patch.start()


def tearDownModule():
    _limits_patch.stop()



def test_progress_counts_every_seat_even_failed_ones(monkeypatch):
    async def fake(spec, messages, **kw):
        if spec == "bad":
            return None
        return {"content": "x"}
    monkeypatch.setattr(openrouter, "query_model", fake)
    progress = {"done": 0, "total": 0}
    asyncio.run(openrouter.query_models_parallel(["a", "bad", "c"], [], progress=progress))
    assert progress == {"done": 3, "total": 3}


def test_progress_total_grows_when_reserves_are_asked(monkeypatch):
    async def fake(spec, messages, **kw):
        return {"content": "x"}
    monkeypatch.setattr(openrouter, "query_model", fake)
    progress = {"done": 0, "total": 0}
    asyncio.run(openrouter.query_models_parallel(["a", "b"], [], progress=progress))
    asyncio.run(openrouter.query_models_parallel(["r"], [], progress=progress))
    assert progress == {"done": 3, "total": 3}
