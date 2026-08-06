"""Constrained candidate expansion utilities."""

from __future__ import annotations

import cv2
import numpy as np

from .detector import _pixel_size_m


def expand_existing_candidates(
    seeds: np.ndarray,
    score: np.ndarray,
    valid: np.ndarray,
    profile: dict,
    *,
    grow_percentile: float,
    max_distance_m: float,
    closing_m: float,
) -> tuple[np.ndarray, float]:
    """Grow each seed only through nearby, score-supported pixels.

    This limits recall expansion to existing candidate locations, avoiding the
    unrelated false positives created by applying a lower threshold globally.
    """

    if not 0 < grow_percentile < 100:
        raise ValueError("grow_percentile must be between 0 and 100.")
    if max_distance_m <= 0:
        raise ValueError("max_distance_m must be positive.")
    values = score[valid]
    if not values.size:
        raise ValueError("The score raster contains no valid pixels.")
    grow_threshold = float(np.percentile(values, grow_percentile))
    growable = ((score >= grow_threshold) & valid).astype(np.uint8)
    component_count, seed_labels = cv2.connectedComponents(seeds.astype(np.uint8), connectivity=8)
    distance_pixels = max(1, round(max_distance_m / _pixel_size_m(profile)))
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * distance_pixels + 1, 2 * distance_pixels + 1))
    expanded = np.zeros(seeds.shape, dtype=np.uint8)
    for component_id in range(1, component_count):
        seed = (seed_labels == component_id).astype(np.uint8)
        neighbourhood = cv2.dilate(seed, dilation_kernel)
        permitted = (growable & neighbourhood).astype(np.uint8)
        _, region_labels = cv2.connectedComponents(permitted, connectivity=8)
        touching = np.unique(region_labels[seed.astype(bool)])
        touching = touching[touching != 0]
        if touching.size:
            expanded[np.isin(region_labels, touching)] = 1
    if closing_m > 0:
        closing_pixels = max(1, round(closing_m / _pixel_size_m(profile)))
        closing_pixels = closing_pixels if closing_pixels % 2 else closing_pixels + 1
        if closing_pixels > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_pixels, closing_pixels))
            expanded = cv2.morphologyEx(expanded, cv2.MORPH_CLOSE, kernel)
    return expanded, grow_threshold
