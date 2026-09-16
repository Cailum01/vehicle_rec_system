import argparse
import json
from pathlib import Path

from .annotation import annotate_video
from .scoring import score_file
from .trajectory import extract_trajectory
from .training import train_segmentation_model


def _prompt_path(message: str) -> str:
    return input(message).strip().strip('"')


def _prompt_int(message: str, default: int) -> int:
    value = input(f"{message} [{default}]: ").strip()
    return default if not value else int(value)


def _run_training_mode(mode: str) -> None:
    if mode == "track":
        print("\n訓練跑道：標註護欄、跑道與得分區")
        video = _prompt_path("跑道影片路徑: ")
        dataset = _prompt_path("訓練資料夾 [data/track_dataset]: ") or "data/track_dataset"
        sample_frames = _prompt_int("每隔幾幀取一張", 10)
        base_model = _prompt_path("基礎模型 [yolo11n-seg.pt]: ") or "yolo11n-seg.pt"
        epochs = _prompt_int("訓練回合數", 100)
        output = _prompt_path("輸出資料夾 [runs/track_training]: ") or "runs/track_training"
        name = "track"
        data_yaml = annotate_video(video, dataset, ["wall", "track", "score_zone"], sample_frames)
        result = train_segmentation_model(data_yaml, output, base_model, epochs, 640, 4, name)
    else:
        print("\n訓練車輛：標註遙控車輪廓")
        video = _prompt_path("車輛影片路徑: ")
        dataset = _prompt_path("訓練資料夾 [data/vehicle_dataset]: ") or "data/vehicle_dataset"
        sample_frames = _prompt_int("每隔幾幀取一張", 10)
        base_model = _prompt_path("基礎模型 [yolo11n-seg.pt]: ") or "yolo11n-seg.pt"
        epochs = _prompt_int("訓練回合數", 100)
        output = _prompt_path("輸出資料夾 [runs/vehicle_training]: ") or "runs/vehicle_training"
        name = "vehicle"
        data_yaml = annotate_video(video, dataset, ["vehicle"], sample_frames)
        result = train_segmentation_model(data_yaml, output, base_model, epochs, 640, 4, name)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _run_score_mode() -> None:
    print("\n評分模式")
    trajectory = _prompt_path("trajectory.json 路徑: ")
    output = _prompt_path("分數輸出路徑 [score.json]: ") or "score.json"
    result = score_file(trajectory, output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def interactive_menu() -> None:
    while True:
        print("\n=== Vehicle Recognition System ===")
        print("1. 評分")
        print("2. 訓練")
        print("0. 離開")
        choice = input("請選擇模式: ").strip()
        if choice == "1":
            _run_score_mode()
        elif choice == "2":
            print("\n--- 訓練模式 ---")
            print("1. 訓練跑道")
            print("2. 訓練車輛")
            print("0. 返回")
            training_choice = input("請選擇訓練項目: ").strip()
            if training_choice == "1":
                _run_training_mode("track")
            elif training_choice == "2":
                _run_training_mode("vehicle")
        elif choice == "0":
            return


def main() -> None:
    parser = argparse.ArgumentParser(description="RC drift trajectory system")
    subcommands = parser.add_subparsers(dest="command")

    extract = subcommands.add_parser("extract", help="extract trajectories and save trajectory.png/json")
    extract.add_argument("--video", required=True)
    extract.add_argument("--output", required=True)
    extract.add_argument("--model", required=True)
    extract.add_argument("--sample-frames", type=int, default=5)
    extract.add_argument("--vehicle-class", type=int, default=0)
    extract.add_argument("--track-model", help="trained track segmentation model")

    score = subcommands.add_parser("score", help="score a saved trajectory without rerunning the model")
    score.add_argument("--trajectory", required=True)
    score.add_argument("--output", required=True)

    track_label = subcommands.add_parser("track-label", help="annotate wall, track and score-zone polygons")
    track_label.add_argument("--video", required=True)
    track_label.add_argument("--dataset", required=True)
    track_label.add_argument("--sample-frames", type=int, default=10)
    track_label.add_argument("--max-frames", type=int)

    vehicle_label = subcommands.add_parser("vehicle-label", help="annotate vehicle polygons")
    vehicle_label.add_argument("--video", required=True)
    vehicle_label.add_argument("--dataset", required=True)
    vehicle_label.add_argument("--sample-frames", type=int, default=10)
    vehicle_label.add_argument("--max-frames", type=int)

    def add_train_arguments(command: argparse.ArgumentParser) -> None:
        command.add_argument("--dataset", required=True, help="dataset data.yaml or dataset directory")
        command.add_argument("--output", required=True)
        command.add_argument("--base-model", default="yolo11n-seg.pt")
        command.add_argument("--epochs", type=int, default=100)
        command.add_argument("--imgsz", type=int, default=640)
        command.add_argument("--batch", type=int, default=4)
        command.add_argument("--name", default="run")

    track_train = subcommands.add_parser("track-train", help="train wall/track/score-zone segmentation model")
    add_train_arguments(track_train)
    vehicle_train = subcommands.add_parser("vehicle-train", help="train vehicle segmentation model")
    add_train_arguments(vehicle_train)

    args = parser.parse_args()
    if args.command is None:
        interactive_menu()
    elif args.command == "extract":
        extract_trajectory(args.video, args.output, args.model, args.sample_frames, args.vehicle_class, args.track_model)
        print(f"Trajectory saved to {Path(args.output).resolve()}")
    elif args.command == "score":
        result = score_file(args.trajectory, args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "track-label":
        data_yaml = annotate_video(args.video, args.dataset, ["wall", "track", "score_zone"], args.sample_frames, args.max_frames)
        print(f"Dataset saved: {data_yaml}")
    elif args.command == "vehicle-label":
        data_yaml = annotate_video(args.video, args.dataset, ["vehicle"], args.sample_frames, args.max_frames)
        print(f"Dataset saved: {data_yaml}")
    elif args.command in ("track-train", "vehicle-train"):
        dataset = Path(args.dataset)
        dataset_yaml = dataset / "data.yaml" if dataset.is_dir() else dataset
        result = train_segmentation_model(dataset_yaml, args.output, args.base_model,
                                          args.epochs, args.imgsz, args.batch, args.name)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
