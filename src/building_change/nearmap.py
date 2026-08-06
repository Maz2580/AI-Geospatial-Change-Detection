"""Nearmap clients for both Staticmap and standard fixed-survey Tile imagery."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
from pathlib import Path
import time
from typing import Iterable

import cv2
import numpy as np
import rasterio
from rasterio.transform import from_origin
import requests


API_ROOT = "https://api.nearmap.com"
WEB_MERCATOR_RADIUS = 6378137.0


class NearmapApiError(RuntimeError):
    """A request to Nearmap failed in a way the caller can act on."""


@dataclass(frozen=True)
class Survey:
    id: str
    capture_date: date
    pixel_sizes: dict[str, float]
    tile_zoom: int | None = None

    @classmethod
    def from_response(cls, value: dict) -> "Survey":
        try:
            pixel_sizes = value.get("pixelSizes") or {}
            if not pixel_sizes and value.get("pixelSize") is not None:
                pixel_sizes = {"Vert": float(value["pixelSize"])}
            return cls(
                id=value["id"],
                capture_date=date.fromisoformat(value["captureDate"][:10]),
                pixel_sizes=pixel_sizes,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise NearmapApiError("Nearmap returned survey metadata in an unexpected format.") from exc

    @classmethod
    def from_tile_response(cls, value: dict, content_type: str = "Vert") -> "Survey":
        survey = cls.from_response(value)
        resources = value.get("resources", {}).get("tiles", [])
        zooms = [tile.get("scale") for tile in resources if tile.get("type") == content_type and tile.get("scale") is not None]
        if not zooms:
            raise NearmapApiError(f"Survey {survey.id} does not expose {content_type} tile metadata.")
        return cls(survey.id, survey.capture_date, survey.pixel_sizes, max(int(zoom) for zoom in zooms))


def parse_date(value: str | date | None) -> date | None:
    if value is None or isinstance(value, date):
        return value
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Expected an ISO date (YYYY-MM-DD), got {value!r}.") from exc


def select_surveys(
    surveys: Iterable[Survey], *, before_date: str | date, after_date: str | date | None = None
) -> tuple[Survey, Survey]:
    """Choose the latest survey on/before before_date and a later survey."""
    before, after = parse_date(before_date), parse_date(after_date)
    ordered = sorted(surveys, key=lambda survey: survey.capture_date)
    if not ordered:
        raise NearmapApiError("No Nearmap surveys were available for this area and content type.")
    older_choices = [survey for survey in ordered if survey.capture_date <= before]
    if not older_choices:
        raise NearmapApiError(f"No survey was captured on or before {before.isoformat()}.")
    older = older_choices[-1]
    if after is None:
        newer = ordered[-1]
    else:
        newer_choices = [survey for survey in ordered if survey.capture_date >= after]
        if not newer_choices:
            raise NearmapApiError(f"No survey was captured on or after {after.isoformat()}.")
        newer = newer_choices[0]
    if newer.capture_date <= older.capture_date or newer.id == older.id:
        raise NearmapApiError("The selected dates resolve to the same (or non-increasing) survey. Choose dates farther apart.")
    return older, newer


class NearmapClient:
    """Retrieve exact-date imagery from Nearmap's Staticmap or Tile APIs."""

    def __init__(self, api_key: str, *, session: requests.Session | None = None, timeout_s: int = 90):
        if not api_key:
            raise ValueError("NEARMAP_API_KEY is missing. Add it to .env before using the Nearmap command.")
        self.api_key = api_key
        self.session = session or requests.Session()
        self.timeout_s = timeout_s

    @property
    def _authorization(self) -> dict[str, str]:
        return {"Authorization": f"Apikey {self.api_key}"}

    def _raise_for_status(self, response: requests.Response) -> None:
        try:
            response.raise_for_status()
        except requests.HTTPError:
            try:
                payload = response.json()
                detail = payload.get("message") or payload.get("error") or ""
            except ValueError:
                detail = response.text[:300].strip()
            message = f"Nearmap returned HTTP {response.status_code}."
            if detail:
                message += f" {detail}"
            if response.status_code == 401:
                message += " Verify that NEARMAP_API_KEY is valid and enabled."
            elif response.status_code == 403:
                message += " Check your subscription and access to this AOI/content type."
            # Do not chain requests' exception: it contains the request URL.
            raise NearmapApiError(message) from None

    def coverage(
        self, *, longitude: float, latitude: float, radius_m: int, resources: Iterable[str]
    ) -> tuple[list[Survey], str]:
        """Staticmap coverage for subscriptions with TrueOrtho/DSM access."""
        if not 1 <= radius_m <= 100:
            raise ValueError("Nearmap Staticmap supports radius_m values from 1 to 100 metres.")
        resource_list = list(dict.fromkeys(resources))
        if not resource_list:
            raise ValueError("At least one Nearmap content type is required.")
        response = self.session.get(
            f"{API_ROOT}/staticmap/v2/coverage.json",
            params={
                "apikey": self.api_key, "point": f"{longitude:.8f},{latitude:.8f}", "radius": radius_m,
                "resources": ",".join(resource_list), "filter": "allTypes", "limit": 1000,
            },
            timeout=self.timeout_s,
        )
        self._raise_for_status(response)
        payload = response.json()
        token = payload.get("transactionToken")
        if not token:
            raise NearmapApiError("Nearmap coverage response did not contain a transaction token.")
        return [Survey.from_response(value) for value in payload.get("surveys", [])], token

    def download(
        self,
        survey: Survey,
        *,
        content_type: str,
        transaction_token: str,
        longitude: float,
        latitude: float,
        radius_m: int,
        size_px: int,
        output_path: str | Path,
    ) -> Path:
        """Download one Staticmap GeoTIFF atomically."""
        if not 1 <= size_px <= 5000:
            raise ValueError("Nearmap Staticmap supports size_px values from 1 to 5000.")
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".part")
        response = self.session.get(
            f"{API_ROOT}/staticmap/v2/surveys/{survey.id}/{content_type}.tif",
            params={
                "transactionToken": transaction_token, "point": f"{longitude:.8f},{latitude:.8f}",
                "radius": radius_m, "size": f"{size_px}x{size_px}",
            },
            timeout=self.timeout_s,
            stream=True,
        )
        self._raise_for_status(response)
        try:
            with temporary.open("wb") as file_handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file_handle.write(chunk)
            temporary.replace(target)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return target

    def tile_coverage(self, *, longitude: float, latitude: float, content_type: str = "Vert") -> list[Survey]:
        """List fixed-date Tile API surveys without the separate Staticmap product."""
        response = self.session.get(
            f"{API_ROOT}/coverage/v2/point/{longitude:.8f},{latitude:.8f}",
            params={"limit": 1000, "resources": f"tiles:{content_type}", "fields": "id,captureDate,pixelSize,resources"},
            headers=self._authorization,
            timeout=self.timeout_s,
        )
        self._raise_for_status(response)
        return [Survey.from_tile_response(value, content_type) for value in response.json().get("surveys", [])]

    @staticmethod
    def _tile_range(longitude: float, latitude: float, radius_m: float, zoom: int) -> tuple[int, int, int, int, float]:
        if not -85.0 < latitude < 85.0:
            raise ValueError("Tile retrieval requires a latitude between -85 and 85 degrees.")
        if not 0 <= zoom <= 30:
            raise ValueError("Tile zoom must be between 0 and 30.")
        world = 2.0 * math.pi * WEB_MERCATOR_RADIUS
        origin = world / 2.0
        count = 1 << zoom
        span = world / count
        center_x = WEB_MERCATOR_RADIUS * math.radians(longitude)
        center_y = WEB_MERCATOR_RADIUS * math.log(math.tan(math.pi / 4.0 + math.radians(latitude) / 2.0))
        # EPSG:3857 map metres are inflated by sec(latitude); convert the requested ground radius.
        map_radius = radius_m / max(math.cos(math.radians(latitude)), 0.1)
        x_start = max(0, math.floor((center_x - map_radius + origin) / span))
        x_end = min(count - 1, math.floor((center_x + map_radius + origin) / span))
        y_start = max(0, math.floor((origin - (center_y + map_radius)) / span))
        y_end = min(count - 1, math.floor((origin - (center_y - map_radius)) / span))
        return x_start, x_end, y_start, y_end, span

    def _get_tile(self, survey: Survey, content_type: str, zoom: int, x: int, y: int) -> np.ndarray:
        url = f"{API_ROOT}/tiles/v3/surveys/{survey.id}/{content_type}/{zoom}/{x}/{y}.jpg"
        for attempt in range(4):
            response = self.session.get(url, headers=self._authorization, timeout=self.timeout_s)
            if response.status_code not in (429, 503) or attempt == 3:
                self._raise_for_status(response)
                image = cv2.imdecode(np.frombuffer(response.content, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None or image.shape[:2] != (256, 256):
                    raise NearmapApiError(f"Nearmap returned an invalid tile at z={zoom}, x={x}, y={y}.")
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            time.sleep(2**attempt)
        raise AssertionError("Tile retry loop unexpectedly ended.")

    def download_tile_mosaic(
        self,
        survey: Survey,
        *,
        longitude: float,
        latitude: float,
        radius_m: int,
        output_path: str | Path,
        content_type: str = "Vert",
        zoom: int | None = None,
        max_tiles: int = 256,
    ) -> Path:
        """Mosaic a small, fixed-date Tile API AOI into a co-registered GeoTIFF."""
        if radius_m <= 0:
            raise ValueError("radius_m must be positive.")
        zoom = zoom if zoom is not None else survey.tile_zoom
        if zoom is None:
            raise NearmapApiError("The survey did not provide a maximum tile zoom level.")
        x_start, x_end, y_start, y_end, tile_span = self._tile_range(longitude, latitude, radius_m, zoom)
        columns, rows = x_end - x_start + 1, y_end - y_start + 1
        tile_count = rows * columns
        if tile_count > max_tiles:
            raise ValueError(f"The requested AOI requires {tile_count} tiles; maximum is {max_tiles}. Reduce radius or zoom.")
        mosaic = np.empty((rows * 256, columns * 256, 3), dtype=np.uint8)
        for row, tile_y in enumerate(range(y_start, y_end + 1)):
            for column, tile_x in enumerate(range(x_start, x_end + 1)):
                mosaic[row * 256:(row + 1) * 256, column * 256:(column + 1) * 256] = self._get_tile(
                    survey, content_type, zoom, tile_x, tile_y
                )
        world = 2.0 * math.pi * WEB_MERCATOR_RADIUS
        origin = world / 2.0
        pixel_size = tile_span / 256.0
        west, north = x_start * tile_span - origin, origin - y_start * tile_span
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".part")
        with rasterio.open(
            temporary, "w", driver="GTiff", width=mosaic.shape[1], height=mosaic.shape[0], count=3,
            dtype="uint8", crs="EPSG:3857", transform=from_origin(west, north, pixel_size, pixel_size), compress="lzw",
        ) as destination:
            destination.write(np.moveaxis(mosaic, -1, 0))
            destination.update_tags(survey_id=survey.id, capture_date=survey.capture_date.isoformat(), tile_zoom=zoom)
        temporary.replace(target)
        return target
