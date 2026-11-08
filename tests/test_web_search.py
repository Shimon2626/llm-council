import unittest
from backend.web_search import resolve_web_search


class WebSearchModeTests(unittest.TestCase):
    def test_auto_on_for_link_and_live_wording(self):
        self.assertTrue(resolve_web_search("auto", "Audit https://example.com/pricing")["enabled"])
        self.assertTrue(resolve_web_search("auto", "What is the latest ruling on this?")["enabled"])

    def test_auto_off_for_drafting_and_math(self):
        self.assertFalse(resolve_web_search("auto", "Draft a strategy memo for the team")["enabled"])
        self.assertFalse(resolve_web_search("auto", "Summarise this text: lorem ipsum")["enabled"])

    def test_url_inside_attachment_does_not_trigger(self):
        q = "Summarise the file\n\n**Attached file: a.txt**\n```\nsee https://example.com\n```"
        self.assertFalse(resolve_web_search("auto", q)["enabled"])

    def test_explicit_modes_win(self):
        self.assertTrue(resolve_web_search("on", "write a poem")["enabled"])
        self.assertFalse(resolve_web_search("off", "https://example.com")["enabled"])
        self.assertFalse(resolve_web_search("bogus", "write a poem")["enabled"])


if __name__ == "__main__":
    unittest.main()
