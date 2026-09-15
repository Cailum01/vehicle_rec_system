import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


WEIGHTS = {
    "route": 40.0,
    "angle": 20.0,
    "speed": 10.0,
    "speed_stability": 10.0,
    "angle_stability": 20.0,
}


def _angle_delta(first: float, second: float) -> float:
    return abs((first - second + 180.0) % 360.0 - 180.0)


def score_track(samples: list[dict[str, Any]], fps: float) -> dict[str, float]:
    if len(samples) < 2:
        return {key: 0.0 for key in (*WEIGHTS, "total")}

    speeds: list[float] = []
    angles: list[float] = []
    for previous, current in zip(samples, samples[1:]):
        frame_delta = current["frame"] - previous["frame"]
        if frame_delta <= 0:
            continue
        dx = current["center"][0] - previous["center"][0]
        dy = current["center"][1] - previous["center"][1]
        speeds.append(math.hypot(dx, dy) * fps / frame_delta)
        movement_angle = math.degrees(math.atan2(dy, dx)) % 360
        vehicle_angle = current.get("angle_degrees")
        if vehicle_angle is not None:
            angles.append(_angle_delta(vehicle_angle, movement_angle))

    speed_mean = mean(speeds) if speeds else 0.0
    speed_variation = pstdev(speeds) if len(speeds) > 1 else 0.0
    angle_variation = pstdev(angles) if len(angles) > 1 else 0.0
    result = {
        "route": WEIGHTS["route"],
        "angle": min(WEIGHTS["angle"], mean(angles) / 45.0 * WEIGHTS["angle"]) if angles else 0.0,
        "speed": min(WEIGHTS["speed"], speed_mean / 100.0 * WEIGHTS["speed"]),
        "speed_stability": max(0.0, WEIGHTS["speed_stability"] * (1.0 - speed_variation / max(speed_mean, 1.0))),
        "angle_stability": max(0.0, WEIGHTS["angle_stability"] * (1.0 - angle_variation / 45.0)),
    }
    result["total"] = min(100.0, max(0.0, sum(result.values())))
    return {key: round(value, 2) for key, value in result.items()}


def score_file(input_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(input_path).read_text(encoding="utf-8"))
    scores = {
        track_id: score_track(samples, data["video_fps"])
        for track_id, samples in data.get("tracks", {}).items()
    }
    result = {"weights": WEIGHTS, "tracks": scores}
    Path(output_path).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
