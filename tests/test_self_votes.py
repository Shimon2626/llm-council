"""Aggregate ranking leaves out each seat's vote for itself (stdlib unittest)."""
import unittest

from backend.council import calculate_aggregate_rankings

L2M = {"Response A": "m/a", "Response B": "m/b", "Response C": "m/c"}


def rank(model, self_label, *order):
    text = "FINAL RANKING:\n" + "\n".join(f"{i}. {o}" for i, o in enumerate(order, 1))
    return {"model": model, "self_label": self_label, "ranking": text}


class SelfVoteTests(unittest.TestCase):
    def setUp(self):
        # A and B each rank themselves first; C ranks A first.
        self.rows = [
            rank("m/a", "Response A", "Response A", "Response B", "Response C"),
            rank("m/b", "Response B", "Response B", "Response A", "Response C"),
            rank("m/c", "Response C", "Response A", "Response B", "Response C"),
        ]

    def agg(self, rows=None):
        return {a["model"]: a for a in calculate_aggregate_rankings(rows or self.rows, L2M)}

    def test_own_vote_is_excluded_from_average_rank(self):
        a = self.agg()["m/a"]
        self.assertEqual(a["average_rank_all_votes"], 1.33)  # 1, 2, 1
        self.assertEqual(a["average_rank"], 1.5)             # 2, 1 (own vote of 1 removed)
        self.assertEqual((a["rankings_count"], a["all_votes_count"], a["self_votes"]), (2, 3, 1))

    def test_both_averages_reported_and_sorted_by_excluding_self(self):
        out = calculate_aggregate_rankings(self.rows, L2M)
        self.assertEqual([a["model"] for a in out], ["m/a", "m/b", "m/c"])
        self.assertTrue(all("average_rank_all_votes" in a for a in out))

    def test_without_self_label_falls_back_to_model_name(self):
        rows = [{k: v for k, v in r.items() if k != "self_label"} for r in self.rows]
        self.assertEqual(self.agg(rows)["m/a"]["average_rank"], 1.5)

    def test_model_ranked_only_by_itself_has_no_external_average(self):
        rows = [rank("m/a", "Response A", "Response A")]
        a = calculate_aggregate_rankings(rows, L2M)[0]
        self.assertIsNone(a["average_rank"]); self.assertEqual(a["average_rank_all_votes"], 1.0)


if __name__ == "__main__":
    unittest.main()
