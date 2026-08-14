"""Place the building boundary using the imagery, not a probability threshold.

The segmentation's probability field is soft: 12.7% of a suburban scene sits in
the ambiguous 0.2--0.8 band, against 14.1% of it being building at all. Choosing
a threshold therefore chooses a boundary position, and moving that threshold from
0.4371 to 0.70 shifts the edge 0.67 m at the median and 4.11 m at p90. The
polygon traces an iso-probability contour, which has no reason to sit on a roof
edge.

The imagery does not share that weakness. A roof edge is a sharp radiometric
discontinuity at 12 cm resolution, and it is in the same place on every date.

So the probability field is used only for what it is reliable at -- saying a
building is *here* -- and the edge is then placed by segmenting the image itself,
seeded from the confident core and the confident background. This is the standard
seeded graph-cut formulation; OpenCV's GrabCut provides it.

Refinement runs per building rather than over the whole scene. A local colour
model separates one roof from its own surroundings far better than a scene-wide
one, which would have to describe every roof material and every garden at once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


class BoundaryRefinementError(ValueError):
    """Raised when boundary refinement cannot run."""


@dataclass(frozen=True)
class BoundaryRefinementConfig:
    """Settings for image-led boundary placement."""

    enabled: bool = False
    # Pixels this confident are treated as building and never revised.
    core_probability: float = 0.75
    # Pixels this unconfident anchor the background colour model.
    background_probability: float = 0.10
    # Context around each building for the local colour model to learn from.
    padding_m: float = 4.0
    # GrabCut converges quickly; more iterations mostly cost time.
    iterations: int = 3
    # Buildings smaller than this are left as the threshold found them: too few
    # pixels to estimate a colour model from.
    min_component_area_m2: float = 8.0

    def validate(self) -> None:
        if not 0 < self.background_probability < self.core_probability < 1:
            raise BoundaryRefinementError(
                "Probabilities must satisfy 0 < background < core < 1."
            )
        if self.padding_m <= 0:
            raise BoundaryRefinementError("padding_m must be positive.")
        if self.iterations < 1:
            raise BoundaryRefinementError("iterations must be at least one.")
        if self.min_component_area_m2 < 0:
            raise BoundaryRefinementError("min_component_area_m2 cannot be negative.")


def _display_rgb(rgb: np.ndarray) -> np.ndarray:
    """Convert a [3, H, W] float array to the uint8 [H, W, 3] GrabCut expects."""
    if rgb.ndim != 3 or rgb.shape[0] != 3:
        raise BoundaryRefinementError("Refinement requires RGB shaped [3, height, width].")
    image = np.nan_to_num(rgb.astype(np.float32), nan=0.0, posinf=255.0, neginf=0.0)
    return np.ascontiguousarray(np.moveaxis(image.clip(0, 255), 0, -1).astype(np.uint8))


def refine_building_mask(
    probability: np.ndarray,
    valid: np.ndarray,
    rgb: np.ndarray,
    *,
    threshold: float,
    pixel_size_m: float,
    config: BoundaryRefinementConfig | None = None,
) -> np.ndarray:
    """Return a building mask whose edges follow the image, not the threshold.

    Each confident core becomes one refinement problem. Pixels above
    ``core_probability`` are fixed as building, pixels below
    ``background_probability`` are fixed as background, and everything between is
    decided by the image. A building the model is unsure about therefore keeps
    its threshold boundary rather than being deleted.
    """
    active = config or BoundaryRefinementConfig()
    active.validate()
    if pixel_size_m <= 0:
        raise BoundaryRefinementError("pixel_size_m must be positive.")
    if probability.shape != valid.shape:
        raise BoundaryRefinementError("Probability and valid masks have different shapes.")

    threshold_mask = (probability >= threshold) & valid
    if not active.enabled or not threshold_mask.any():
        return threshold_mask

    image = _display_rgb(rgb)
    if image.shape[:2] != probability.shape:
        raise BoundaryRefinementError("Imagery and probability raster have different shapes.")

    core = (probability >= active.core_probability) & valid
    count, labels, stats, _ = cv2.connectedComponentsWithStats(core.astype(np.uint8), connectivity=8)
    pixel_area = pixel_size_m * pixel_size_m
    padding_px = max(1, int(round(active.padding_m / pixel_size_m)))
    height, width = probability.shape

    # Which threshold-mask blob each core sits in. A refined core *replaces* its
    # blob rather than adding to it, so refinement can pull an over-large
    # boundary inward. Retaining the blob as well would make the result a union
    # of the two, which can only ever grow.
    _, threshold_labels = cv2.connectedComponents(threshold_mask.astype(np.uint8), connectivity=8)

    refined = np.zeros_like(threshold_mask)
    replaced_labels: set[int] = set()

    for component in range(1, count):
        left, top, box_width, box_height, pixel_count = stats[component]
        if pixel_count * pixel_area < active.min_component_area_m2:
            continue
        row_start = max(0, top - padding_px)
        row_stop = min(height, top + box_height + padding_px)
        column_start = max(0, left - padding_px)
        column_stop = min(width, left + box_width + padding_px)
        window = (slice(row_start, row_stop), slice(column_start, column_stop))

        window_probability = probability[window]
        window_valid = valid[window]
        window_core = labels[window] == component

        mask = np.full(window_probability.shape, cv2.GC_BGD, dtype=np.uint8)
        mask[window_probability >= active.background_probability] = cv2.GC_PR_BGD
        mask[(window_probability >= threshold) & window_valid] = cv2.GC_PR_FGD
        mask[window_core] = cv2.GC_FGD
        mask[~window_valid] = cv2.GC_BGD

        # GrabCut needs both classes present to estimate two colour models.
        if not (mask == cv2.GC_FGD).any() or not (mask == cv2.GC_BGD).any():
            continue
        try:
            cv2.grabCut(
                np.ascontiguousarray(image[window]),
                mask,
                None,
                np.zeros((1, 65), np.float64),
                np.zeros((1, 65), np.float64),
                active.iterations,
                cv2.GC_INIT_WITH_MASK,
            )
        except cv2.error:  # pragma: no cover - depends on the OpenCV build
            continue

        decided = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)) & window_valid
        # Keep only the part connected to this component's core: GrabCut can
        # annex a neighbouring roof of similar colour across a shared boundary.
        decided = _component_containing(decided, window_core)
        if not decided.any():
            continue
        refined[window] |= decided
        replaced_labels.update(int(label) for label in np.unique(threshold_labels[window][window_core]) if label)

    # A blob whose core was refined takes the new edge. A blob with no usable
    # core -- too small or too uncertain to model -- keeps its threshold edge, so
    # refinement never loses a detection it cannot improve.
    kept = threshold_mask & ~np.isin(threshold_labels, list(replaced_labels)) if replaced_labels else threshold_mask
    return (refined | kept) & valid


def _component_containing(mask: np.ndarray, seed: np.ndarray) -> np.ndarray:
    """Keep only mask components that overlap the seed."""
    if not mask.any() or not seed.any():
        return mask
    count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return mask
    keep = set(np.unique(labels[seed & mask])) - {0}
    if not keep:
        return mask
    return np.isin(labels, list(keep))


def refinement_report(
    threshold_mask: np.ndarray, refined_mask: np.ndarray, pixel_size_m: float
) -> dict[str, Any]:
    """Describe how far refinement moved the boundary, for the run record."""
    pixel_area = pixel_size_m * pixel_size_m
    added = int((refined_mask & ~threshold_mask).sum())
    removed = int((threshold_mask & ~refined_mask).sum())
    return {
        "threshold_area_m2": round(float(threshold_mask.sum() * pixel_area), 1),
        "refined_area_m2": round(float(refined_mask.sum() * pixel_area), 1),
        "area_added_m2": round(float(added * pixel_area), 1),
        "area_removed_m2": round(float(removed * pixel_area), 1),
    }
