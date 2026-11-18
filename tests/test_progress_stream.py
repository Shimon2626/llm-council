import asyncio
import json

from fastapi.testclient import TestClient

from backend import main, storage


def test_stream_emits_real_progress_events(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DATA_DIR", str(tmp_path))

    async def s1(q, models, mt, cost_log, web, progress):
        progress.update(total=2)
        await asyncio.sleep(0.1)
        progress["done"] = 1
        await asyncio.sleep(0.3)
        progress["done"] = 2
        return [{"model": "m", "seat": "m", "response": "r", "status": "ok"}]

    async def s2(q, s1r, models, mt, cost_log, progress):
        progress.update(total=1, done=1)
        await asyncio.sleep(0.3)
        return [], {}

    async def s3(*a, **k):
        return {"model": "c", "response": "final"}

    monkeypatch.setattr(main, "stage1_collect_responses", s1)
    monkeypatch.setattr(main, "stage2_collect_rankings", s2)
    monkeypatch.setattr(main, "stage3_synthesize_final", s3)

    orig = main._run_with_heartbeat
    def fast(coro, request, **kw):
        return orig(coro, request, poll_interval=0.1, **kw)
    monkeypatch.setattr(main, "_run_with_heartbeat", fast)

    client = TestClient(main.app)
    conv = client.post("/api/conversations", json={}).json()
    with client.stream("POST", f"/api/conversations/{conv['id']}/message/stream", json={"content": "hello"}) as r:
        events = [json.loads(l[6:]) for l in r.iter_lines() if l.startswith("data: ")]
    prog = [(e["stage"], e["done"], e["total"]) for e in events if e["type"] == "progress"]
    assert (1, 2, 2) in prog and any(p[0] == 2 for p in prog)
    assert events[-1]["type"] in ("complete", "cost_complete", "title_complete")
