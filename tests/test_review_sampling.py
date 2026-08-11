from __future__ import annotations

import unittest

from building_change.review_sampling import (
    SampleConfig,
    SamplingError,
    draw_sample,
    estimate_precision,
    size_band,
    stratum_of,
)


def _feature(candidate_id: int, area_m2: float, tier: str) -> dict:
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
        "properties": {
            "candidate_id": candidate_id,
            "area_m2": area_m2,
            "change_support": {"tier": tier},
        },
    }


def _population() -> dict:
    features = []
    identifier = 1
    for tier, count in [("corroborated", 6), ("weakly_corroborated", 68), ("footprint_only", 68)]:
        for index in range(count):
            area = [30.0, 100.0, 300.0][index % 3]
            features.append(_feature(identifier, area, tier))
            identifier += 1
    return {"type": "FeatureCollection", "features": features}


class SizeBandTests(unittest.TestCase):
    def test_bands_partition_the_range(self) -> None:
        self.assertEqual(size_band(10), "small")
        self.assertEqual(size_band(59.9), "small")
        self.assertEqual(size_band(60), "medium")
        self.assertEqual(size_band(149.9), "medium")
        self.assertEqual(size_band(150), "large")
        self.assertEqual(size_band(10_000), "large")

    def test_stratum_combines_tier_and_size(self) -> None:
        self.assertEqual(stratum_of(_feature(1, 200, "corroborated")), "corroborated|large")


class DrawSampleTests(unittest.TestCase):
    def test_sample_is_deterministic_for_a_fixed_seed(self) -> None:
        first, _ = draw_sample(_population(), SampleConfig(seed=7))
        second, _ = draw_sample(_population(), SampleConfig(seed=7))

        ids = lambda s: [f["properties"]["candidate_id"] for f in s["features"]]
        self.assertEqual(ids(first), ids(second))

    def test_every_stratum_is_represented(self) -> None:
        sample, plan = draw_sample(_population(), SampleConfig(target_size=25))

        drawn = {f["properties"]["review_stratum"] for f in sample["features"]}
        self.assertEqual(drawn, set(plan["strata"]))
        for info in plan["strata"].values():
            self.assertGreaterEqual(info["sampled"], 1)

    def test_never_samples_more_than_a_stratum_holds(self) -> None:
        population = {"type": "FeatureCollection", "features": [_feature(1, 300, "corroborated")]}

        sample, plan = draw_sample(population, SampleConfig(target_size=25, min_per_stratum=5))

        self.assertEqual(len(sample["features"]), 1)
        self.assertEqual(plan["strata"]["corroborated|large"]["sampled"], 1)

    def test_weights_reconstruct_the_population(self) -> None:
        _, plan = draw_sample(_population(), SampleConfig(target_size=25))

        rebuilt = sum(info["sampled"] * info["weight"] for info in plan["strata"].values())
        self.assertAlmostEqual(rebuilt, plan["population"], delta=1.0)

    def test_review_order_runs_largest_first(self) -> None:
        sample, _ = draw_sample(_population(), SampleConfig(target_size=25))

        orders = [f["properties"]["review_order"] for f in sample["features"]]
        areas = [f["properties"]["area_m2"] for f in sample["features"]]
        self.assertEqual(orders, sorted(orders))
        self.assertEqual(areas, sorted(areas, reverse=True))

    def test_rejects_empty_or_malformed_input(self) -> None:
        with self.assertRaises(SamplingError):
            draw_sample({"type": "FeatureCollection", "features": []})
        with self.assertRaises(SamplingError):
            draw_sample({"type": "Feature"})

    def test_rejects_invalid_configuration(self) -> None:
        with self.assertRaises(SamplingError):
            SampleConfig(target_size=0).validate()
        with self.assertRaises(SamplingError):
            SampleConfig(min_per_stratum=0).validate()


class EstimatePrecisionTests(unittest.TestCase):
    def test_weights_strata_by_population_not_by_sample_size(self) -> None:
        population = {
            "type": "FeatureCollection",
            "features": [_feature(i, 300.0, "corroborated") for i in range(1, 3)]
            + [_feature(i, 30.0, "footprint_only") for i in range(3, 99)],
        }
        sample, plan = draw_sample(population, SampleConfig(target_size=10, seed=1))

        # Everything in the tiny stratum is right, everything in the big one is wrong.
        labels = {}
        for feature in sample["features"]:
            stratum = feature["properties"]["review_stratum"]
            labels[str(feature["properties"]["candidate_id"])] = (
                "correct" if stratum.startswith("corroborated") else "wrong"
            )

        result = estimate_precision(labels, plan, sample)

        # An unweighted average would sit far higher; population weighting keeps it low.
        self.assertLess(result["weighted_precision"], 0.1)

    def test_ignores_unlabelled_candidates(self) -> None:
        sample, plan = draw_sample(_population(), SampleConfig(target_size=25))
        first = sample["features"][0]["properties"]["candidate_id"]

        result = estimate_precision({str(first): "correct"}, plan, sample)

        self.assertEqual(result["labelled"], 1)
        self.assertEqual(result["weighted_precision"], 1.0)

    def test_returns_none_when_nothing_is_labelled(self) -> None:
        sample, plan = draw_sample(_population(), SampleConfig(target_size=25))

        result = estimate_precision({}, plan, sample)

        self.assertIsNone(result["weighted_precision"])


if __name__ == "__main__":
    unittest.main()
