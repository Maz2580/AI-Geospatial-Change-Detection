"""Ground measurement in true metres.

Web Mercator is a projected CRS, so ``crs.is_projected`` is true for it, but its
units are not ground metres: area is inflated by sec(latitude)^2, which is 1.605x
at Melbourne and 1.541x at Murchison. Every Nearmap tile mosaic this project
downloads is EPSG:3857, so any area or distance measured in place is wrong by
more than half again.

The downloader already knew this -- ``nearmap.py`` converts a requested ground
radius into map metres before choosing tiles -- but the measurement code did not,
and the two never met. This module is the single place that knows the
difference, so they cannot diverge again.

Areas come from the ellipsoid by geodesic integration, which needs no projection
and no zone choice. Distances, intersections and unions need a plane, so they use
the local UTM zone: within one small area of interest its scale factor is
effectively constant, leaving at most about 0.2% residual against the 60% error
it replaces.
"""

from __future__ import annotations

import math
from typing import Any

from pyproj import CRS, Geod, Transformer
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform

WGS84 = "EPSG:4326"
WEB_MERCATOR = "EPSG:3857"

# GDA2020 / MGA zone 55. Pinned for the Victorian gold set so a benchmark number
# is reproducible regardless of where a future area of interest sits.
VICTORIAN_METRIC_CRS = "EPSG:7855"

_GEOD = Geod(ellps="WGS84")


class GeodesyError(ValueError):
    """Raised when geometry cannot be measured on the ground."""


def utm_epsg_for(longitude: float, latitude: float) -> int:
    """Return the WGS84 UTM EPSG code containing a coordinate.

    WGS84 UTM rather than a national grid: the source geometry is already WGS84,
    so this is a projection with no datum shift to get wrong.
    """
    if not -180.0 <= longitude <= 180.0 or not -90.0 <= latitude <= 90.0:
        raise GeodesyError(f"Coordinate out of range: {longitude}, {latitude}")
    zone = int(math.floor((longitude + 180.0) / 6.0) % 60) + 1
    return (32700 if latitude < 0 else 32600) + zone


def metric_crs_for(geometry: BaseGeometry | dict[str, Any]) -> CRS:
    """Return a local projected CRS in true metres for WGS84 geometry."""
    resolved = shape(geometry) if isinstance(geometry, dict) else geometry
    if resolved.is_empty:
        raise GeodesyError("Cannot choose a metric CRS for empty geometry.")
    point = resolved.representative_point()
    return CRS.from_epsg(utm_epsg_for(point.x, point.y))


def to_metric(geometry: BaseGeometry | dict[str, Any], crs: Any = None) -> tuple[BaseGeometry, CRS]:
    """Project WGS84 geometry into true ground metres.

    Passing ``crs`` keeps a set of geometries in one plane, which is required
    before comparing or combining them.
    """
    resolved = shape(geometry) if isinstance(geometry, dict) else geometry
    target = CRS.from_user_input(crs) if crs is not None else metric_crs_for(resolved)
    transformer = Transformer.from_crs(WGS84, target, always_xy=True).transform
    projected = shapely_transform(transformer, resolved)
    if not projected.is_valid:
        projected = projected.buffer(0)
    if projected.is_empty:
        raise GeodesyError("Geometry became empty when projected to a metric CRS.")
    return projected, target


def from_metric(geometry: BaseGeometry, crs: Any) -> dict[str, Any]:
    """Return WGS84 GeoJSON for geometry measured in a metric CRS."""
    transformer = Transformer.from_crs(CRS.from_user_input(crs), WGS84, always_xy=True).transform
    return mapping(shapely_transform(transformer, geometry))


def geodesic_area_m2(geometry: BaseGeometry | dict[str, Any]) -> float:
    """Return the true ground area of WGS84 geometry, in square metres.

    Computed on the ellipsoid, so it is exact rather than as good as the chosen
    projection, and it needs no zone selection near a zone boundary.
    """
    resolved = shape(geometry) if isinstance(geometry, dict) else geometry
    if resolved.is_empty:
        return 0.0
    area, _ = _GEOD.geometry_area_perimeter(resolved)
    return abs(float(area))


def ground_area_m2(geometry: BaseGeometry, source_crs: Any) -> float:
    """Return true ground area for geometry expressed in ``source_crs``.

    Replaces the assumption that any projected CRS measures in ground metres.
    Web Mercator satisfies ``is_projected`` and does not.
    """
    if geometry.is_empty:
        return 0.0
    if source_crs is None:
        raise GeodesyError("Geometry needs a CRS before its area means anything.")
    crs = CRS.from_user_input(source_crs)
    if crs.is_geographic:
        return geodesic_area_m2(geometry)
    transformer = Transformer.from_crs(crs, WGS84, always_xy=True).transform
    return geodesic_area_m2(shapely_transform(transformer, geometry))


def web_mercator_area_inflation(latitude: float) -> float:
    """Factor by which EPSG:3857 overstates area at a latitude.

    Kept explicit so the size of the historical error can be quoted when
    comparing against figures produced before this module existed.
    """
    if not -85.0 < latitude < 85.0:
        raise GeodesyError("Web Mercator is undefined beyond 85 degrees.")
    return 1.0 / math.cos(math.radians(latitude)) ** 2
