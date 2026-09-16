from __future__ import annotations

from pathlib import Path
from typing import Any

from ultralytics import YOLO


def train_segmentation_model(
    dataset_yaml: str | Path,
    output_dir: str | Path,
    base_model: str = "yolo11n-seg.pt",
    epochs: int = 100,
    image_size: int = 640,
    batch: int = 4,
    name: str = "run",
) -> dict[str, Any]:
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if image_size < 32:
        raise ValueError("image_size must be at least 32")
    if batch < 1:
        raise ValueError("batch must be at least 1")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model = YOLO(base_model)
    results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=image_size,
        batch=batch,
        project=str(output),
        name=name,
    )
    save_dir = Path(getattr(results, "save_dir", output / name))
    best_model = save_dir / "weights" / "best.pt"
    return {
        "save_dir": str(save_dir),
        "best_model": str(best_model),
        "exists": best_model.exists(),
    }
