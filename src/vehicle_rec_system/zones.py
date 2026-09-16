from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def load_zone_config(path: str | Path) -> dict[str, Any]:
    import json

    return json.loads(Path(path).read_text(encoding="utf-8"))


def point_in_polygons(point: list[int] | tuple[int, int], polygons: list[list[list[float]]]) -> bool:
    x, y = point
    return any(cv2.pointPolygonTest(np.asarray(polygon, dtype=np.float32), (x, y), False) >= 0 for polygon in polygons)


def mask_zone_state(point: list[int] | tuple[int, int], polygons_by_class: dict[int, list[list[list[float]]]]) -> dict[str, bool]:
    return {
        "in_track": point_in_polygons(point, polygons_by_class.get(1, [])),
        "in_score_zone": point_in_polygons(point, polygons_by_class.get(2, [])),
        "near_wall": point_in_polygons(point, polygons_by_class.get(0, [])),
    }


def polygons_from_result(result: Any) -> dict[int, list[list[list[float]]]]:
    polygons_by_class: dict[int, list[list[list[float]]]] = {}
    if result.masks is None or result.boxes is None:
        return polygons_by_class
    classes = result.boxes.cls.int().cpu().tolist()
    for class_id, polygon in zip(classes, result.masks.xy):
        polygons_by_class.setdefault(int(class_id), []).append(polygon.tolist())
    return polygons_by_class
