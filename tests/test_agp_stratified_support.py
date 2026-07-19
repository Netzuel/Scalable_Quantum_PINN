from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from agp_stratified_support import stratified_ranked_selection  # noqa: E402


def candidate(label: str, score: float) -> dict[str, object]:
    return {"label": label, "score": score}


class AGPStratifiedSupportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            candidate("XIIIIIII", 10.0),
            candidate("IXIIIIII", 9.0),
            candidate("IIIIIIXI", 8.0),
            candidate("IIIIIIIX", 7.0),
            candidate("XXIIIIII", 6.0),
            candidate("IXXIIIII", 5.0),
            candidate("IIIIIXXI", 4.0),
            candidate("IIIIIIXX", 3.0),
        ]

    def test_exact_budget_no_duplicates_and_deterministic_output(self):
        rows = self.rows + [candidate("XIIIIIII", 100.0)]
        first = stratified_ranked_selection(
            rows,
            6,
            q=8,
            locality_quotas={"1": 2, "2": 2},
            spatial_bins=2,
            seed=17,
        )
        second = stratified_ranked_selection(
            rows,
            6,
            q=8,
            locality_quotas={"1": 2, "2": 2},
            spatial_bins=2,
            seed=17,
        )

        labels = [row["label"] for row in first.selected_rows]
        self.assertEqual(len(labels), 6)
        self.assertEqual(len(set(labels)), 6)
        self.assertEqual(labels, [row["label"] for row in second.selected_rows])
        self.assertEqual(first.provenance, second.provenance)

    def test_locality_quotas_are_spread_over_available_spatial_bins(self):
        selection = stratified_ranked_selection(
            self.rows,
            4,
            q=8,
            locality_quotas={"1": 2, "2": 2},
            spatial_bins=2,
            seed=0,
        )

        labels = [str(row["label"]) for row in selection.selected_rows]
        orders = [sum(symbol != "I" for symbol in label) for label in labels]
        centers = [
            sum(index for index, symbol in enumerate(label) if symbol != "I")
            / sum(symbol != "I" for symbol in label)
            for label in labels
        ]
        self.assertEqual(orders.count(1), 2)
        self.assertEqual(orders.count(2), 2)
        self.assertTrue(any(center < 4 for center in centers))
        self.assertTrue(any(center >= 4 for center in centers))
        self.assertEqual(selection.provenance["realized_locality_counts"], {"1": 2, "2": 2})

    def test_empty_stratum_quota_is_redistributed_by_global_ranking(self):
        selection = stratified_ranked_selection(
            self.rows[:4],
            3,
            q=8,
            locality_quotas={"1": 1, "3+": 2},
            spatial_bins=2,
            seed=0,
        )

        self.assertEqual(
            [row["label"] for row in selection.selected_rows],
            ["XIIIIIII", "IXIIIIII", "IIIIIIXI"],
        )
        self.assertEqual(selection.provenance["realized_locality_counts"]["3+"], 0)
        self.assertEqual(selection.provenance["selected_terms"], 3)

    def test_budget_larger_than_available_candidates_is_clipped(self):
        selection = stratified_ranked_selection(
            self.rows[:2],
            10,
            q=8,
            locality_quotas={"1": 4},
            spatial_bins=4,
            seed=3,
        )

        self.assertEqual(len(selection.selected_rows), 2)
        self.assertEqual(selection.provenance["requested_terms"], 10)
        self.assertEqual(selection.provenance["selected_terms"], 2)


if __name__ == "__main__":
    unittest.main()
