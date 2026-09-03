"""Validate .ork files for the egg_payload / ballast MassComponent convention (§4.1)."""

from __future__ import annotations

import zipfile
from pathlib import Path

EGG_COMPONENT_NAME = "egg_payload"
BALLAST_COMPONENT_NAME = "ballast"


class OrkValidationError(ValueError):
    """Raised when a .ork file is missing required named components."""


def _ork_text_blob(ork_path: Path) -> str:
    """Read searchable text from a .ork (plain XML or ZIP-wrapped)."""
    raw = ork_path.read_bytes()
    if raw[:2] == b"PK":
        with zipfile.ZipFile(ork_path) as zf:
            chunks: list[str] = []
            for name in zf.namelist():
                try:
                    chunks.append(zf.read(name).decode("utf-8", errors="ignore"))
                except Exception:
                    continue
            return "\n".join(chunks)
    return raw.decode("utf-8", errors="ignore")


def required_mass_component_names() -> tuple[str, str]:
    return EGG_COMPONENT_NAME, BALLAST_COMPONENT_NAME


def validate_ork_mass_components(ork_path: str | Path) -> list[str]:
    """Return list of missing required component names (empty = OK)."""
    path = Path(ork_path)
    if not path.is_file():
        raise FileNotFoundError(path)

    blob = _ork_text_blob(path)
    missing: list[str] = []
    for name in required_mass_component_names():
        if name not in blob:
            missing.append(name)
    return missing


def assert_ork_mass_components(ork_path: str | Path) -> None:
    missing = validate_ork_mass_components(ork_path)
    if missing:
        raise OrkValidationError(
            f"{ork_path} is missing required MassComponent name(s): {missing}. "
            f"Pre-place components named {list(required_mass_component_names())} "
            "at fixed stations per design spec §4.1."
        )
