from pathlib import Path

from vehicle_rec_system.annotation import Polygon, _write_label
from vehicle_rec_system.scoring import score_track
from vehicle_rec_system.zones import mask_zone_state


def test_polygon_label_is_normalized(tmp_path: Path):
    label_path = tmp_path / "frame.txt"
    _write_label(label_path, [Polygon(0, [(0, 0), (50, 0), (50, 100)])], 100, 200)
    assert label_path.read_text(encoding="utf-8") == "0 0.000000 0.000000 0.500000 0.000000 0.500000 0.500000\n"


def test_route_state_and_score_use_track_polygon():
    polygons = {1: [[[0, 0], [100, 0], [100, 100], [0, 100]]], 2: [], 0: []}
    assert mask_zone_state([50, 50], polygons)["in_track"] is True
    assert mask_zone_state([150, 50], polygons)["in_track"] is False

    samples = [
        {"frame": 0, "center": [50, 50], "in_track": True},
        {"frame": 5, "center": [150, 50], "in_track": False},
    ]
    result = score_track(samples, fps=30)
    assert result["route"] == 20.0
