"""Draw a stratified review sample so a small labelling effort yields real numbers.

Labelling all 142 candidates is a chore, and labelling the top 25 by size only
measures the easy end. This draws a stratified sample across support tier and
size band, so the resulting precision estimate covers the whole distribution
rather than just the obvious cases.

Each stratum's precision is estimated from its own sample, then recombined
weighted by stratum size. That keeps the overall estimate unbiased even though
small candidates are deliberately over-sampled relative to their share.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import random
from typing import Any

SIZE_BANDS: tuple[tuple[str, float, float], ...] = (
    ("small", 0.0, 60.0),
    ("medium", 60.0, 150.0),
    ("large", 150.0, float("inf")),
)


class SamplingError(ValueError):
    """Raised when a review sample cannot be drawn."""


@dataclass(frozen=True)
class SampleConfig:
    """How many candidates to draw and how to spread them."""

    target_size: int = 25
    seed: int = 20260811
    min_per_stratum: int = 2

    def validate(self) -> None:
        if self.target_size < 1:
            raise SamplingError("target_size must be at least one.")
        if self.min_per_stratum < 1:
            raise SamplingError("min_per_stratum must be at least one.")


def size_band(area_m2: float) -> str:
    for name, low, high in SIZE_BANDS:
        if low <= area_m2 < high:
            return name
    return SIZE_BANDS[-1][0]


def stratum_of(feature: dict[str, Any]) -> str:
    properties = feature.get("properties", {})
    support = properties.get("change_support") or {}
    tier = support.get("tier", "unknown")
    return f"{tier}|{size_band(float(properties.get('area_m2', 0.0)))}"


def draw_sample(
    candidates: dict[str, Any],
    config: SampleConfig | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a stratified subset plus the weights needed to extrapolate from it."""
    active = config or SampleConfig()
    active.validate()

    features = candidates.get("features")
    if candidates.get("type") != "FeatureCollection" or not isinstance(features, list):
        raise SamplingError("Candidates must be a GeoJSON FeatureCollection.")
    if not features:
        raise SamplingError("There are no candidates to sample.")

    strata: dict[str, list[dict[str, Any]]] = {}
    for feature in features:
        strata.setdefault(stratum_of(feature), []).append(feature)

    rng = random.Random(active.seed)
    total = len(features)
    picked: list[dict[str, Any]] = []
    weights: dict[str, Any] = {}

    for name, members in sorted(strata.items()):
        share = len(members) / total
        wanted = max(active.min_per_stratum, round(active.target_size * share))
        wanted = min(wanted, len(members))
        chosen = rng.sample(members, wanted)
        picked.extend(chosen)
        weights[name] = {
            "population": len(members),
            "sampled": wanted,
            # One reviewed candidate stands for this many in the full set.
            "weight": round(len(members) / wanted, 3),
        }

    for order, feature in enumerate(sorted(picked, key=lambda f: -float(f["properties"].get("area_m2", 0.0))), start=1):
        feature.setdefault("properties", {})["review_order"] = order
        feature["properties"]["review_stratum"] = stratum_of(feature)

    sample = {
        "type": "FeatureCollection",
        "features": sorted(picked, key=lambda f: f["properties"]["review_order"]),
        "metadata": {
            "sampled_from": total,
            "sample_size": len(picked),
            "seed": active.seed,
            "strata": weights,
        },
    }
    plan = {
        "population": total,
        "sample_size": len(picked),
        "strata": weights,
        "note": (
            "Estimate precision per stratum, then recombine weighted by population share. "
            "Small candidates are over-sampled on purpose, so an unweighted average would be misleading."
        ),
    }
    return sample, plan


def estimate_precision(labels: dict[str, str], plan: dict[str, Any], sample: dict[str, Any]) -> dict[str, Any]:
    """Combine per-stratum hit rates into a weighted precision estimate.

    ``labels`` maps candidate_id (as a string) to "correct" or anything else.
    """
    per_stratum: dict[str, dict[str, Any]] = {}
    for feature in sample.get("features", []):
        properties = feature.get("properties", {})
        stratum = properties.get("review_stratum", "unknown")
        verdict = labels.get(str(properties.get("candidate_id")))
        if verdict is None:
            continue
        entry = per_stratum.setdefault(stratum, {"labelled": 0, "correct": 0})
        entry["labelled"] += 1
        entry["correct"] += 1 if verdict == "correct" else 0

    population = plan.get("population", 0)
    weighted_correct = 0.0
    covered = 0
    for stratum, entry in per_stratum.items():
        info = plan.get("strata", {}).get(stratum)
        if not info or not entry["labelled"]:
            continue
        rate = entry["correct"] / entry["labelled"]
        entry["precision"] = round(rate, 4)
        weighted_correct += rate * info["population"]
        covered += info["population"]

    return {
        "labelled": sum(e["labelled"] for e in per_stratum.values()),
        "population_covered": covered,
        "population": population,
        "weighted_precision": round(weighted_correct / covered, 4) if covered else None,
        "per_stratum": per_stratum,
    }
