from backend import storage


def test_run_record_survives_save_and_reload(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DATA_DIR", str(tmp_path))
    conv = storage.create_conversation("c1")
    storage.add_assistant_message(
        conv["id"], [], [], {"model": "m", "response": "r"}, None,
        {"label_to_model": {}, "aggregate_rankings": [], "run_record": {"typed_prompt": "hi"}},
    )
    saved = storage.get_conversation(conv["id"])["messages"][-1]["metadata"]
    assert saved["run_record"] == {"typed_prompt": "hi"}
