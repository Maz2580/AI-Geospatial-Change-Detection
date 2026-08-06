"""Convenient, GIS-environment-safe entry point for the change workflow."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


def _prefer_rasterio_projection_data() -> None:
    """Avoid an incompatible external OSGeo PROJ_LIB in this process only."""
    spec = importlib.util.find_spec("rasterio")
    if spec is None or spec.origin is None:
        return
    bundled_data = Path(spec.origin).parent / "proj_data"
    if bundled_data.is_dir():
        os.environ["PROJ_DATA"] = str(bundled_data)
        os.environ.pop("PROJ_LIB", None)


_prefer_rasterio_projection_data()

from building_change.cli import main  # noqa: E402  (must follow environment setup)


if __name__ == "__main__":
    main()
