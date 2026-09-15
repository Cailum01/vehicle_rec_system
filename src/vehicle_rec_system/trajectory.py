import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from .models import TrajectoryData


def extract_trajectory(video_path: str | Path, output_dir: str | Path, model_path: str,
                       sample_frames: int = 5, vehicle_class: int | None = 0) -> TrajectoryData:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    data = TrajectoryData(fps, width, height, max(1, sample_frames))
    model = YOLO(model_path)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    frame_index = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame_index += 1
        if frame_index % data.sample_frames:
            continue
        classes = None if vehicle_class is None else [vehicle_class]
        result = model.track(frame, persist=True, classes=classes, verbose=False)[0]
        if result.boxes is None or result.boxes.id is None:
            continue
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        track_ids = result.boxes.id.int().cpu().numpy()
        for index, track_id in enumerate(track_ids):
            x1, y1, x2, y2 = boxes[index].tolist()
            center = [(x1 + x2) // 2, (y1 + y2) // 2]
            sample = {"frame": frame_index, "time_seconds": frame_index / fps,
                      "track_id": int(track_id), "bbox": [x1, y1, x2, y2], "center": center}
            data.tracks.setdefault(str(int(track_id)), []).append(sample)
            points = data.tracks[str(int(track_id))]
            if len(points) > 1:
                cv2.line(canvas, tuple(points[-2]["center"]), tuple(center), (0, 220, 255), 3)
            cv2.circle(canvas, tuple(center), 5, (0, 255, 0), -1)

    capture.release()
    (output / "trajectory.json").write_text(json.dumps(data.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "metadata.json").write_text(json.dumps({"video": str(video_path), "fps": fps,
                                                        "width": width, "height": height,
                                                        "sample_frames": sample_frames}, indent=2), encoding="utf-8")
    cv2.imwrite(str(output / "trajectory.png"), canvas)
    return data
