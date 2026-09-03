"""Load CalibrationModel artifacts (§3.5 / §4.2 registry layout)."""

from __future__ import annotations

import json
from pathlib import Path

from optimizer.schemas.calibration import CalibrationModel


def load_calibration_file(path: str | Path) -> CalibrationModel:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return CalibrationModel.model_validate(data)


def resolve_latest_calibration(
    calibration_dir: str | Path,
    rocket_id: str,
) -> Path:
    """Resolve ``calibration/<rocket_id>/latest.json`` (pointer or full artifact)."""
    root = Path(calibration_dir)
    # Accept either calibration_dir == repo calibration/ or .../calibration/<rocket_id>
    candidates = [
        root / rocket_id / "latest.json",
        root / "latest.json",
        root / rocket_id / "latest.json",
    ]
    for path in candidates:
        if path.is_file():
            # latest.json may be a pointer {"path": "v1_....json"} or a full model
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if isinstance(raw, dict) and "path" in raw and "corrections" not in raw:
                pointed = path.parent / str(raw["path"])
                if pointed.is_file():
                    return pointed
            return path
    raise FileNotFoundError(
        f"No calibration latest.json for rocket_id={rocket_id!r} under {root}"
    )


def load_calibration_for_rocket(
    rocket_id: str,
    *,
    calibration_file: str | Path | None = None,
    calibration_dir: str | Path | None = None,
) -> CalibrationModel:
    if calibration_file is not None:
        return load_calibration_file(calibration_file)
    if calibration_dir is not None:
        return load_calibration_file(resolve_latest_calibration(calibration_dir, rocket_id))
    return CalibrationModel.identity(rocket_id)
