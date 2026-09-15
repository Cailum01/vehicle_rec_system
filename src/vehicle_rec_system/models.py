from dataclasses import dataclass, field
from typing import Any


@dataclass
class Detection:
    frame: int
    time_seconds: float
    track_id: int
    bbox: list[int]
    center: list[int]
    angle_degrees: float | None = None


@dataclass
class TrajectoryData:
    video_fps: float
    frame_width: int
    frame_height: int
    sample_frames: int
    tracks: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_fps": self.video_fps,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "sample_frames": self.sample_frames,
            "tracks": self.tracks,
        }
