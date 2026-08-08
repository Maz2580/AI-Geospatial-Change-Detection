"""Validation-manifest support for model comparison without model training.

The project has useful human-reviewed evidence, but it is not all complete
ground truth.  This module locks the current pilot evidence by hash and makes
its permitted use explicit before any new pretrained model is compared.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class ValidationManifestError(ValueError):
    """Raised when the validation manifest is malformed or has changed."""


def repository_root() -> Path:
    """Return the repository root when this package is imported from source."""
    return Path(__file__).resolve().parents[2]


def load_validation_manifest(path: str | Path) -> dict[str, Any]:
    """Read a validation manifest without changing it."""
    source = Path(path)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationManifestError(f"Could not read validation manifest: {source}") from exc
    if not isinstance(document, dict):
        raise ValidationManifestError("Validation manifest must be a JSON object.")
    return document


def _relative_artifact_path(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValidationManifestError("Each validation artifact needs a non-empty relative path.")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValidationManifestError("Validation artifact paths must stay inside the repository.")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def validate_manifest(document: dict[str, Any], *, root: str | Path | None = None) -> dict[str, Any]:
    """Validate locked pilot evidence and return a compact audit report.

    This deliberately validates only committed labels and references. Imagery
    exports and model predictions are runtime artifacts, so their absence must
    never make a clean clone fail validation.
    """
    if document.get("schema_version") != 1:
        raise ValidationManifestError("Validation manifest schema_version must be 1.")
    if not isinstance(document.get("benchmark"), str) or not document["benchmark"]:
        raise ValidationManifestError("Validation manifest needs a benchmark name.")
    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValidationManifestError("Validation manifest must contain at least one case.")

    base = Path(root) if root is not None else repository_root()
    try:
        base = base.resolve(strict=True)
    except OSError as exc:
        raise ValidationManifestError(f"Validation repository root does not exist: {base}") from exc

    seen_case_ids: set[str] = set()
    verified_artifacts = 0
    report_cases: list[dict[str, Any]] = []
    for case in cases:
        if not isinstance(case, dict):
            raise ValidationManifestError("Each validation case must be an object.")
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValidationManifestError("Each validation case needs a case_id.")
        if case_id in seen_case_ids:
            raise ValidationManifestError(f"Duplicate validation case_id: {case_id}")
        seen_case_ids.add(case_id)
        if case.get("split") != "pilot":
            raise ValidationManifestError(f"Validation case {case_id} must currently be a pilot case.")
        capabilities = case.get("supported_measurements")
        limitations = case.get("limitations")
        artifacts = case.get("artifacts")
        if not isinstance(capabilities, list) or not capabilities or not all(isinstance(value, str) and value for value in capabilities):
            raise ValidationManifestError(f"Validation case {case_id} needs supported_measurements.")
        if not isinstance(limitations, list) or not limitations or not all(isinstance(value, str) and value for value in limitations):
            raise ValidationManifestError(f"Validation case {case_id} needs limitations.")
        if not isinstance(artifacts, list) or not artifacts:
            raise ValidationManifestError(f"Validation case {case_id} needs at least one locked artifact.")

        verified_paths: list[str] = []
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                raise ValidationManifestError(f"Validation case {case_id} has an invalid artifact.")
            relative_path = _relative_artifact_path(artifact.get("path"))
            expected_hash = artifact.get("sha256")
            if not isinstance(expected_hash, str) or len(expected_hash) != 64:
                raise ValidationManifestError(f"Validation artifact {relative_path} needs a SHA-256 hash.")
            source = (base / relative_path).resolve()
            if base not in source.parents:
                raise ValidationManifestError(f"Validation artifact escapes repository: {relative_path}")
            if not source.is_file():
                raise ValidationManifestError(f"Locked validation artifact is missing: {relative_path}")
            actual_hash = _sha256(source)
            if actual_hash != expected_hash.upper():
                raise ValidationManifestError(
                    f"Locked validation artifact changed: {relative_path}. "
                    "Create a new benchmark revision instead of silently changing pilot evidence."
                )
            verified_artifacts += 1
            verified_paths.append(relative_path.as_posix())
        report_cases.append(
            {
                "case_id": case_id,
                "split": "pilot",
                "supported_measurements": capabilities,
                "limitations": limitations,
                "verified_artifacts": verified_paths,
            }
        )

    return {
        "benchmark": document["benchmark"],
        "schema_version": 1,
        "case_count": len(report_cases),
        "verified_artifact_count": verified_artifacts,
        "cases": report_cases,
        "warning": "All current cases are pilot evidence. They are frozen for comparison, but they are not sufficient to certify a production model or to tune a production threshold.",
    }
