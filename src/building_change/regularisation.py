"""Dominant-orientation regularisation for extracted building footprints.

Segmentation masks vectorise into staircase polygons that follow pixel edges
rather than building edges.  This wraps ``buildingregulariser`` (MIT, authored
by WA DPIRD) to snap each polygon's edges to its own dominant orientation, and
optionally to align neighbouring buildings onto a shared estate/street grid.

Regularisation is a batch operation: neighbour alignment needs to see the whole
candidate set at once, so callers should collect geometries first and pass them
in together rather than regularising one polygon at a time.

**It is off by default, because measured against human-drawn roofs it makes the
outlines worse.**  It was adopted on an edge-angle measurement -- error fell from
15.87 deg to 4.97 deg -- which describes angular consistency, not position.  The
regulariser assumes its input boundary is roughly right and merely ragged, so it
estimates a dominant orientation from the polygon and snaps edges to it.  When
the input is a probability-threshold contour that is already systematically
wrong, that orientation is estimated from the error and snapping propagates it
along edges that had been closer to correct.  The result looks more like a
building and sits further from one.

Turn it back on when the boundary reaching it is known to be close, or when a
tidy outline matters more than an accurate one.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Sequence

from shapely.geometry.base import BaseGeometry

logger = logging.getLogger(__name__)

# The library's own guidance: simplify tolerance should be 2-3x the raster GSD.
_TOLERANCE_PIXEL_MULTIPLIER = 2.5


class RegularisationError(ValueError):
    """Raised when footprint regularisation is configured but cannot run."""


@dataclass(frozen=True)
class RegularisationConfig:
    """Settings for polygon regularisation.

    Off by default. Measured against 80 human-drawn roofs in the UC5 gold set,
    regularisation lowered boundary F1 at 0.25 m from 0.478 to 0.340 and raised
    the number of buildings missed entirely -- at every threshold tested. See
    ``docs/outline_accuracy.md``.
    """

    enabled: bool = False
    simplify_tolerance_m: float = 0.2
    allow_45_degree: bool = True
    align_neighbours: bool = True
    neighbour_search_distance_m: float = 100.0
    # Defaults to serial: the library uses multiprocessing.Pool, which needs a
    # __main__ guard in every caller on Windows spawn.
    num_cores: int = 1

    def validate(self) -> None:
        if self.simplify_tolerance_m <= 0:
            raise RegularisationError("simplify_tolerance_m must be positive.")
        if self.neighbour_search_distance_m <= 0:
            raise RegularisationError("neighbour_search_distance_m must be positive.")
        if self.num_cores < 1:
            raise RegularisationError("num_cores must be at least one.")


def tolerance_for_pixel_size(pixel_size_m: float) -> float:
    """Return the recommended simplify tolerance for a given ground sample distance."""
    if pixel_size_m <= 0:
        raise RegularisationError("pixel_size_m must be positive.")
    return round(pixel_size_m * _TOLERANCE_PIXEL_MULTIPLIER, 4)


def regularise_geometries(
    geometries: Sequence[BaseGeometry],
    crs: Any,
    config: RegularisationConfig | None = None,
) -> list[BaseGeometry]:
    """Regularise footprint polygons expressed in a projected CRS.

    Returns geometries positionally aligned with the input.  A polygon that the
    regulariser drops or cannot process is passed through unchanged, so callers
    never silently lose a candidate.
    """
    active_config = config or RegularisationConfig()
    active_config.validate()

    if not active_config.enabled or not geometries:
        return list(geometries)

    if crs is None or not getattr(crs, "is_projected", False):
        logger.warning("Skipping regularisation: a projected CRS is required, got %s.", crs)
        return list(geometries)

    try:
        import geopandas as gpd
        from buildingregulariser import regularize_geodataframe
    except ImportError as exc:
        raise RegularisationError(
            "Footprint regularisation needs geopandas and buildingregulariser. "
            "Install them with: pip install -r requirements.txt"
        ) from exc

    frame = gpd.GeoDataFrame(
        {"_source_index": range(len(geometries))},
        geometry=list(geometries),
        crs=crs,
    )
    regularised = regularize_geodataframe(
        frame,
        simplify_tolerance=active_config.simplify_tolerance_m,
        allow_45_degree=active_config.allow_45_degree,
        neighbor_alignment=active_config.align_neighbours,
        neighbor_search_distance=active_config.neighbour_search_distance_m,
        num_cores=active_config.num_cores,
    )

    # The regulariser may split one input into several parts; keep the largest.
    result = list(geometries)
    for source_index, group in regularised.groupby("_source_index"):
        candidate = max(group.geometry, key=lambda geometry: geometry.area)
        if candidate.is_empty:
            continue
        if not candidate.is_valid:
            candidate = candidate.buffer(0)
        if candidate.is_empty:
            continue
        result[int(source_index)] = candidate
    return result
