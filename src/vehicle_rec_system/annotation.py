from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np


@dataclass
class Polygon:
    class_id: int
    points: list[tuple[int, int]]


class PolygonAnnotator:
    def __init__(self, window_name: str, class_names: list[str]) -> None:
        self.window_name = window_name
        self.class_names = class_names
        self.active_class = 0
        self.current_points: list[tuple[int, int]] = []
        self.polygons: list[Polygon] = []
        self.saved_count = 0
        self.frame_width = 1
        self.frame_height = 1
        self.display_width = 1
        self.display_height = 1
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1600, 900)
        cv2.setMouseCallback(window_name, self._on_mouse)

    def _on_mouse(self, event: int, x: int, y: int, flags: int, userdata: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            image_x = round((x - self.offset_x) / self.scale)
            image_y = round((y - self.offset_y) / self.scale)
            if 0 <= image_x < self.frame_width and 0 <= image_y < self.frame_height:
                self.current_points.append((image_x, image_y))

    def render(self, frame):
        self.frame_height, self.frame_width = frame.shape[:2]
        self.scale = min(1600 / self.frame_width, 780 / self.frame_height)
        self.display_width = max(1, round(self.frame_width * self.scale))
        self.display_height = max(1, round(self.frame_height * self.scale))
        self.offset_x = (1600 - self.display_width) // 2
        self.offset_y = 80 + (780 - self.display_height) // 2

        canvas = frame.copy()
        colors = [(0, 220, 255), (0, 180, 0), (255, 120, 0), (220, 0, 220)]
        for polygon in self.polygons:
            color = colors[polygon.class_id % len(colors)]
            points = polygon.points
            if len(points) >= 3:
                cv2.polylines(canvas, [self._as_array(points)], True, color, 3)
            for point in points:
                cv2.circle(canvas, point, 5, color, -1)
            label_point = points[0] if points else (0, 0)
            cv2.putText(canvas, self.class_names[polygon.class_id], label_point,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        if self.current_points:
            cv2.polylines(canvas, [self._as_array(self.current_points)], False, (255, 255, 255), 2)
            for point in self.current_points:
                cv2.circle(canvas, point, 5, (255, 255, 255), -1)

        resized = cv2.resize(canvas, (self.display_width, self.display_height), interpolation=cv2.INTER_AREA)
        display = np.full((900, 1600, 3), (35, 35, 35), dtype=np.uint8)
        display[self.offset_y:self.offset_y + self.display_height,
            self.offset_x:self.offset_x + self.display_width] = resized

        cv2.rectangle(display, (0, 0), (1600, 80), (0, 0, 0), -1)
        cv2.putText(display, "VEHICLE-REC ANNOTATION", (18, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.85, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(display, "1/2/3: CLASS     LEFT CLICK: DRAW     ENTER: FINISH POLYGON",
                (18, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.rectangle(display, (0, 850), (1600, 900), (0, 0, 0), -1)
        status = f"CLASS: {self.class_names[self.active_class]}   SAVED: {self.saved_count}"
        cv2.putText(display, status, (18, 883), cv2.FONT_HERSHEY_SIMPLEX,
                0.72, (0, 255, 0), 2, cv2.LINE_AA)
        help_text = "N: SAVE     U: UNDO     C: CLEAR     S: SKIP     Q: QUIT"
        cv2.putText(display, help_text, (610, 883), cv2.FONT_HERSHEY_SIMPLEX,
                0.62, (0, 255, 255), 2, cv2.LINE_AA)
        return display

    @staticmethod
    def _as_array(points: list[tuple[int, int]]):
        return np.asarray([points], dtype=np.int32)

    def handle_key(self, key: int) -> str:
        if key in (ord("q"), ord("Q")):
            return "quit"
        if key in (ord("s"), ord("S")):
            self.reset()
            return "skip"
        if key in (ord("u"), ord("U")):
            if self.current_points:
                self.current_points.pop()
            elif self.polygons:
                self.polygons.pop()
            return "continue"
        if key in (ord("c"), ord("C")):
            self.reset()
            return "continue"
        if key in (ord("1"), ord("2"), ord("3"), ord("4")):
            class_id = int(chr(key)) - 1
            if class_id < len(self.class_names):
                self.active_class = class_id
            return "continue"
        if key in (10, 13) and len(self.current_points) >= 3:
            self.polygons.append(Polygon(self.active_class, self.current_points.copy()))
            self.current_points.clear()
            return "continue"
        if key in (ord("n"), ord("N")):
            if self.current_points and len(self.current_points) >= 3:
                self.polygons.append(Polygon(self.active_class, self.current_points.copy()))
            return "save"
        return "continue"

    def reset(self) -> None:
        self.current_points.clear()
        self.polygons.clear()

    def set_saved_count(self, saved_count: int) -> None:
        self.saved_count = saved_count


def annotate_video(
    video_path: str | Path,
    dataset_dir: str | Path,
    class_names: list[str],
    sample_frames: int = 10,
    max_frames: int | None = None,
    on_saved: Callable[[int], None] | None = None,
) -> Path:
    if sample_frames < 1:
        raise ValueError("sample_frames must be at least 1")

    dataset = Path(dataset_dir)
    images_dir = dataset / "images" / "train"
    labels_dir = dataset / "labels" / "train"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    annotator = PolygonAnnotator("vehicle-rec annotation", class_names)
    saved = 0
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok or (max_frames is not None and saved >= max_frames):
                break
            frame_index += 1
            if (frame_index - 1) % sample_frames:
                continue

            annotator.reset()
            while True:
                cv2.imshow(annotator.window_name, annotator.render(frame))
                key = cv2.waitKey(30) & 0xFF
                action = annotator.handle_key(key)
                if action in ("save", "skip", "quit"):
                    break
            if action == "quit":
                break
            if action == "skip":
                continue
            if not annotator.polygons:
                continue

            image_name = f"frame_{frame_index:08d}.jpg"
            cv2.imwrite(str(images_dir / image_name), frame)
            _write_label(labels_dir / f"frame_{frame_index:08d}.txt", annotator.polygons, frame.shape[1], frame.shape[0])
            saved += 1
            annotator.set_saved_count(saved)
            if on_saved:
                on_saved(saved)
    finally:
        capture.release()
        cv2.destroyAllWindows()

    data_yaml = dataset / "data.yaml"
    data_yaml.write_text(
        "path: .\n"
        "train: images/train\n"
        "val: images/train\n"
        f"nc: {len(class_names)}\n"
        f"names: {json.dumps(class_names, ensure_ascii=False)}\n",
        encoding="utf-8",
    )
    return data_yaml


def _write_label(path: Path, polygons: list[Polygon], width: int, height: int) -> None:
    lines = []
    for polygon in polygons:
        normalized = [f"{x / width:.6f} {y / height:.6f}" for x, y in polygon.points]
        lines.append(f"{polygon.class_id} {' '.join(normalized)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
