"""Preflight: attachment table, and a stop when the MANIFEST names a missing file."""
import hashlib
import unittest

from fastapi.testclient import TestClient

from backend import main
from backend.preflight import build_preflight, manifest_entries, missing_message

PLAN_SHA = "c6458d8a0cc8a3c2504d9cf3f9b914a8ca9177a6742d517dc2eba0e6cf789adf"
AUDIT_SHA = "a" * 64
PROMPT_DOC = f"""[Extracted text from Council_Prompt.docx]

Council Prompt: Table Optimization Plan
MANIFEST
File
Bytes
SHA-256
Version
Table_Optimization_Plan.docx
241632
{PLAN_SHA}
v1_0
eea3014d-attachment.txt
5120
{AUDIT_SHA}
v1_0
STANDING PREAMBLE
You are a reviewer.
"""


def block(name, text):
    return f"**Attached file: {name}**\n```\n{text}\n```"


class ManifestTests(unittest.TestCase):
    def test_reads_names_and_hashes_one_cell_per_line(self):
        e = manifest_entries(PROMPT_DOC)
        self.assertEqual([x["name"] for x in e], ["Table_Optimization_Plan.docx", "eea3014d-attachment.txt"])
        self.assertEqual(e[0]["sha256"], PLAN_SHA)

    def test_no_manifest_means_nothing_to_check(self):
        self.assertEqual(manifest_entries("just a question"), [])


class PreflightTests(unittest.TestCase):
    content = "\n\n".join(["", block("Table_Optimization_Plan.docx", "plan text"), block("Council_Prompt.docx", PROMPT_DOC)])

    def test_missing_manifest_file_stops_the_run_and_names_it(self):
        r = build_preflight(self.content)
        self.assertFalse(r["ok"])
        self.assertEqual([m["name"] for m in r["missing"]], ["eea3014d-attachment.txt"])
        self.assertIn("eea3014d-attachment.txt", missing_message(r))

    def test_table_has_name_chars_and_hash_for_every_attachment(self):
        r = build_preflight(self.content)
        self.assertEqual([x["name"] for x in r["rows"]], ["Table_Optimization_Plan.docx", "Council_Prompt.docx"])
        self.assertEqual(r["rows"][0]["text_sha256"], hashlib.sha256(b"plan text").hexdigest())
        self.assertEqual(r["rows"][0]["status"], "accepted")

    def test_empty_typed_prompt_is_reported(self):
        self.assertEqual(build_preflight(self.content)["typed_prompt_chars"], 0)

    def test_skipped_file_is_listed_with_its_reason_and_explains_the_gap(self):
        r = build_preflight(self.content, [{"name": "eea3014d-attachment.txt", "bytes": 5120,
                                             "status": "skipped", "reason": "looks like a binary file"}])
        row = next(x for x in r["rows"] if x["name"] == "eea3014d-attachment.txt")
        self.assertEqual((row["status"], row["reason"]), ("skipped", "looks like a binary file"))
        self.assertIn("looks like a binary file", r["missing"][0]["why"])

    def test_all_files_attached_passes_and_a_different_hash_warns(self):
        content = self.content + "\n\n" + block("eea3014d-attachment.txt", "audit")
        files = [{"name": "eea3014d-attachment.txt", "bytes": 5, "sha256": "b" * 64}]
        r = build_preflight(content, files)
        self.assertTrue(r["ok"]); self.assertEqual(r["missing"], [])
        self.assertTrue(any("differs from the MANIFEST" in w for w in r["warnings"]))


class EndpointTests(unittest.TestCase):
    def test_preflight_endpoint_returns_table_and_message(self):
        res = TestClient(main.app).post("/api/preflight", json={"content": PreflightTests.content})
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertFalse(body["ok"]); self.assertIn("not attached", body["message"])

    def test_sending_is_refused_before_any_seat_is_called(self):
        with self.assertRaises(Exception) as cm:
            main._refuse_if_manifest_file_missing(PreflightTests.content)
        self.assertEqual(cm.exception.status_code, 422)
        self.assertIn("eea3014d-attachment.txt", cm.exception.detail)


if __name__ == "__main__":
    unittest.main()
