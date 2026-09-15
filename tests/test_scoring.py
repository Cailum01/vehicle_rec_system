from vehicle_rec_system.scoring import score_track


def test_score_track_returns_bounded_weighted_score():
    samples = [
        {"frame": 0, "center": [0, 0], "angle_degrees": 45},
        {"frame": 5, "center": [50, 0], "angle_degrees": 45},
        {"frame": 10, "center": [100, 0], "angle_degrees": 45},
    ]
    result = score_track(samples, fps=30)
    assert 0 <= result["total"] <= 100
    assert result["route"] == 40.0
    assert result["angle"] >= 0
