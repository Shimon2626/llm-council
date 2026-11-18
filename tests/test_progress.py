import asyncio
from backend import openrouter


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
