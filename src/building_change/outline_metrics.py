"""Measure how closely a predicted building outline follows a human-drawn one.

Every accuracy figure in this project so far has described *whether* a change was
found, never *where its edge sits*.  The regularisation work measured edge angle
consistency (15.87 deg -> 4.97 deg), which says nothing about position: an outline a
metre off the roof can still have perfectly parallel edges.

The metrics here are deliberately boundary-first.  Intersection-over-union
saturates on large roofs -- a 600 m2 warehouse outline can be a metre out
everywhere and still score 0.93 -- so boundary F1 at an explicit tolerance is the
primary number, with IoU kept for comparability with published work.

All geometry is measured in a local projected CRS.  Web Mercator inflates area by
sec(latitude)^2, which is 1.60x at Melbourne's latitude, so measuring in EPSG:3857
would overstate every building by 60%.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

# GDA2020 / MGA zone 55: the projected CRS for Victoria, metres, minimal distortion.
VICTORIAN_METRIC_CRS = "EPSG:7855"

# Tolerances span the range that matters operationally: 0.25 m is roughly two
# pixels of 12 cm imagery, and 2.0 m is the point at which an outline is no longer
# describing the same roof edge.
DEFAULT_TOLERANCES_M: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)


class OutlineMetricError(ValueError):
    """Raised when outline geometry cannot be measured."""


@dataclass(frozen=True)
class OutlineScore:
    """Boundary and area agreement between one prediction and one label."""

    iou: float
    boundary_f1: dict[float, float]
    hausdorff_95_m: float
    area_error_fraction: float
    predicted_area_m2: float
    label_area_m2: float
    predicted_vertices: int
    label_vertices: int


def _polygons(geometry: BaseGeometry) -> list[Polygon]:
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)
    raise OutlineMetricError(f"Expected a Polygon or MultiPolygon, got {geometry.geom_type}.")


def vertex_count(geometry: BaseGeometry) -> int:
    """Count exterior and interior ring vertices."""
    return sum(
        len(polygon.exterior.coords) + sum(len(ring.coords) for ring in polygon.interiors)
        for polygon in _polygons(geometry)
    )


def _boundary_points(geometry: BaseGeometry, spacing_m: float) -> np.ndarray:
    """Sample a polygon's boundary at a fixed ground spacing.

    Densifying by distance rather than using the stored vertices is essential
    here: a regularised 12-vertex rectangle and a 340-vertex staircase describe
    the same edge, and vertex-based sampling would score them differently for a
    reason that has nothing to do with accuracy.
    """
    samples: list[tuple[float, float]] = []
    for polygon in _polygons(geometry):
        for ring in [polygon.exterior, *polygon.interiors]:
            length = ring.length
            if length <= 0:
                continue
            count = max(2, int(np.ceil(length / spacing_m)))
            for distance in np.linspace(0.0, length, count, endpoint=False):
                point = ring.interpolate(float(distance))
                samples.append((point.x, point.y))
    if not samples:
        raise OutlineMetricError("Geometry has no boundary to sample.")
    return np.asarray(samples)


def _nearest_distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return each source point's distance to the closest target point."""
    # Chunked so a warehouse boundary does not allocate a huge dense matrix.
    result = np.empty(len(source), dtype=np.float64)
    chunk = max(1, int(2_000_000 / max(1, len(target))))
    for start in range(0, len(source), chunk):
        block = source[start : start + chunk]
        deltas = block[:, None, :] - target[None, :, :]
        result[start : start + len(block)] = np.sqrt((deltas**2).sum(axis=2)).min(axis=1)
    return result


def boundary_f1(
    predicted: BaseGeometry,
    label: BaseGeometry,
    tolerance_m: float,
    *,
    spacing_m: float = 0.25,
) -> float:
    """Fraction of boundary within ``tolerance_m``, as a symmetric F1.

    Precision is the share of the predicted boundary close to the label;
    recall is the share of the label boundary close to the prediction. Both are
    needed: a prediction that traces only one wall of a house scores high
    precision and low recall.
    """
    if tolerance_m <= 0:
        raise OutlineMetricError("tolerance_m must be positive.")
    predicted_points = _boundary_points(predicted, spacing_m)
    label_points = _boundary_points(label, spacing_m)
    precision = float((_nearest_distances(predicted_points, label_points) <= tolerance_m).mean())
    recall = float((_nearest_distances(label_points, predicted_points) <= tolerance_m).mean())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def hausdorff_95(
    predicted: BaseGeometry,
    label: BaseGeometry,
    *,
    spacing_m: float = 0.25,
) -> float:
    """95th-percentile symmetric boundary distance in metres.

    The 95th percentile rather than the maximum, because a single spurious spur
    -- which these outlines have in quantity -- would otherwise dominate an
    otherwise accurate outline's score.
    """
    predicted_points = _boundary_points(predicted, spacing_m)
    label_points = _boundary_points(label, spacing_m)
    forward = _nearest_distances(predicted_points, label_points)
    backward = _nearest_distances(label_points, predicted_points)
    return float(np.percentile(np.concatenate([forward, backward]), 95))


def intersection_over_union(predicted: BaseGeometry, label: BaseGeometry) -> float:
    union = predicted.union(label).area
    return float(predicted.intersection(label).area / union) if union > 0 else 0.0


def score_pair(
    predicted: BaseGeometry,
    label: BaseGeometry,
    *,
    tolerances_m: Sequence[float] = DEFAULT_TOLERANCES_M,
    spacing_m: float = 0.25,
) -> OutlineScore:
    """Score one matched prediction/label pair. Both must be in a metric CRS."""
    if predicted.is_empty or label.is_empty:
        raise OutlineMetricError("Cannot score an empty geometry.")
    label_area = float(label.area)
    predicted_area = float(predicted.area)
    return OutlineScore(
        iou=intersection_over_union(predicted, label),
        boundary_f1={
            float(tolerance): boundary_f1(predicted, label, float(tolerance), spacing_m=spacing_m)
            for tolerance in tolerances_m
        },
        hausdorff_95_m=hausdorff_95(predicted, label, spacing_m=spacing_m),
        area_error_fraction=(predicted_area - label_area) / label_area if label_area > 0 else float("nan"),
        predicted_area_m2=predicted_area,
        label_area_m2=label_area,
        predicted_vertices=vertex_count(predicted),
        label_vertices=vertex_count(label),
    )


# ---------------------------------------------------------------------------
# Instance matching
# ---------------------------------------------------------------------------
#
# Correspondence uses overlap of the *smaller* shape, not IoU and not centroid
# containment. The reasons are specific to what fails here:
#
#   IoU >= 0.5 is the published convention, but it turns a delineation error into
#   a detection error. A 40 m2 shed found correctly and outlined 1 m too large
#   scores IoU 0.45, and is then recorded as both a miss and a false alarm --
#   hiding the outline problem as two detection problems.
#
#   Centroid containment separates detection from delineation properly, but it
#   breaks on the two shapes that matter most here: an L-shaped or courtyard roof
#   whose centroid falls outside itself, and a blob covering many units, whose
#   centroid lands in whichever unit happens to sit in the middle.
#
#   Overlap of the smaller shape handles both. When a blob swallows eleven units,
#   each unit is >= 50% inside the blob, so all eleven correspond and the merge is
#   visible. When one roof is split across three predictions, each prediction is
#   wholly inside the label, so all three correspond and the split is visible.
#
# Splits and merges are reported separately rather than folded into the counts.
# The gold set exists because a previous metric hid exactly this.

# Half of the smaller shape: enough that a badly offset outline still corresponds
# to the building it is describing, strict enough that a neighbouring roof does not.
DEFAULT_MIN_OVERLAP_FRACTION = 0.5


@dataclass(frozen=True)
class MatchResult:
    """Correspondence between predicted outlines and human labels."""

    # One-to-one best pairs, for scoring outline quality: (prediction, label).
    matches: list[tuple[int, int]]
    # Labels no prediction corresponds to: genuine misses.
    unmatched_labels: list[int]
    # Predictions no label corresponds to: genuine false alarms.
    unmatched_predictions: list[int]
    # Labels corresponded to by more than one prediction: one roof split up.
    split_labels: list[int]
    # Predictions corresponding to more than one label: separate roofs merged.
    merged_predictions: list[int]
    # Every correspondence, before the one-to-one reduction.
    correspondences: list[tuple[int, int]]


def match_instances(
    predictions: Sequence[BaseGeometry],
    labels: Sequence[BaseGeometry],
    *,
    min_overlap_fraction: float = DEFAULT_MIN_OVERLAP_FRACTION,
) -> MatchResult:
    """Decide which predicted outlines correspond to which human labels.

    A prediction and a label correspond when their intersection covers at least
    ``min_overlap_fraction`` of whichever of the two is smaller. Correspondence is
    many-to-many by design, so merges and splits survive into the report instead
    of being flattened into hit-and-miss counts.

    ``matches`` reduces those correspondences to one-to-one pairs, best IoU
    first, and is what the boundary metrics should be computed over: scoring a
    blob against all eleven roofs it covers would count its one boundary eleven
    times.
    """
    if not 0 < min_overlap_fraction <= 1:
        raise OutlineMetricError("min_overlap_fraction must be above zero and at most one.")

    correspondences: list[tuple[int, int]] = []
    ranked: list[tuple[float, int, int]] = []
    for prediction_index, prediction in enumerate(predictions):
        if prediction.is_empty:
            continue
        for label_index, label in enumerate(labels):
            if label.is_empty:
                continue
            overlap = prediction.intersection(label).area
            if overlap <= 0:
                continue
            smaller = min(prediction.area, label.area)
            if smaller <= 0 or overlap / smaller < min_overlap_fraction:
                continue
            correspondences.append((prediction_index, label_index))
            ranked.append((intersection_over_union(prediction, label), prediction_index, label_index))

    predictions_per_label: dict[int, int] = {}
    labels_per_prediction: dict[int, int] = {}
    for prediction_index, label_index in correspondences:
        predictions_per_label[label_index] = predictions_per_label.get(label_index, 0) + 1
        labels_per_prediction[prediction_index] = labels_per_prediction.get(prediction_index, 0) + 1

    matches: list[tuple[int, int]] = []
    claimed_predictions: set[int] = set()
    claimed_labels: set[int] = set()
    for _, prediction_index, label_index in sorted(ranked, key=lambda item: -item[0]):
        if prediction_index in claimed_predictions or label_index in claimed_labels:
            continue
        matches.append((prediction_index, label_index))
        claimed_predictions.add(prediction_index)
        claimed_labels.add(label_index)

    return MatchResult(
        matches=sorted(matches),
        unmatched_labels=[index for index in range(len(labels)) if index not in predictions_per_label],
        unmatched_predictions=[index for index in range(len(predictions)) if index not in labels_per_prediction],
        split_labels=sorted(index for index, count in predictions_per_label.items() if count > 1),
        merged_predictions=sorted(index for index, count in labels_per_prediction.items() if count > 1),
        correspondences=sorted(correspondences),
    )


def merge_touching(geometries: Iterable[BaseGeometry]) -> BaseGeometry:
    """Union geometries, used to compare against a 'one connected roof' label."""
    return unary_union(list(geometries))


def summarise(scores: Sequence[OutlineScore], tolerances_m: Sequence[float] = DEFAULT_TOLERANCES_M) -> dict[str, Any]:
    """Aggregate per-instance scores into a reportable summary.

    Medians rather than means: a single 6,000 m2 warehouse would otherwise
    dominate a residential chip's figure.
    """
    if not scores:
        return {"instance_count": 0}
    return {
        "instance_count": len(scores),
        "median_iou": round(float(np.median([score.iou for score in scores])), 4),
        "median_boundary_f1": {
            str(tolerance): round(
                float(np.median([score.boundary_f1[float(tolerance)] for score in scores])), 4
            )
            for tolerance in tolerances_m
        },
        "median_hausdorff_95_m": round(float(np.median([score.hausdorff_95_m for score in scores])), 3),
        "median_area_error_fraction": round(
            float(np.median([score.area_error_fraction for score in scores])), 4
        ),
        "median_predicted_vertices": int(np.median([score.predicted_vertices for score in scores])),
        "median_label_vertices": int(np.median([score.label_vertices for score in scores])),
    }
