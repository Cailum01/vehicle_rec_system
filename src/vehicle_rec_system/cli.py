import argparse
from pathlib import Path

from .scoring import score_file
from .trajectory import extract_trajectory


def main() -> None:
    parser = argparse.ArgumentParser(description="RC drift trajectory system")
    subcommands = parser.add_subparsers(dest="command", required=True)

    extract = subcommands.add_parser("extract", help="extract trajectories and save trajectory.png/json")
    extract.add_argument("--video", required=True)
    extract.add_argument("--output", required=True)
    extract.add_argument("--model", required=True)
    extract.add_argument("--sample-frames", type=int, default=5)
    extract.add_argument("--vehicle-class", type=int, default=0)

    score = subcommands.add_parser("score", help="score a saved trajectory without rerunning the model")
    score.add_argument("--trajectory", required=True)
    score.add_argument("--output", required=True)

    args = parser.parse_args()
    if args.command == "extract":
        extract_trajectory(args.video, args.output, args.model, args.sample_frames, args.vehicle_class)
        print(f"Trajectory saved to {Path(args.output).resolve()}")
    elif args.command == "score":
        result = score_file(args.trajectory, args.output)
        print(result)


if __name__ == "__main__":
    main()
