import argparse
import os

from src.inference import DEFAULT_MODEL_ID
from src.webcam_pipeline import run_webcam_pipeline


def _env_or_default(name: str, default, cast):
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return cast(value)
    except ValueError:
        return default


def parse_args():
    parser = argparse.ArgumentParser(description="Run the webcam classifier")
    parser.add_argument("--model-path", default=os.getenv("MODEL_PATH"), help="Local directory of cached model files")
    parser.add_argument("--model-id", default=os.getenv("MODEL_ID", DEFAULT_MODEL_ID), help="HF model id (used if model-path not set)")
    parser.add_argument("--camera-index", type=int, default=_env_or_default("CAMERA_INDEX", 0, int), help="Camera device index")
    parser.add_argument(
        "--decision-seconds",
        type=float,
        default=_env_or_default("DECISION_SECONDS", 7.0, float),
        help="Seconds before forcing a decision prompt",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=_env_or_default("CONFIDENCE_THRESHOLD", 0.95, float),
        help="Confidence required to trigger prompt early",
    )
    parser.add_argument(
        "--history-size",
        type=int,
        default=_env_or_default("HISTORY_SIZE", 30, int),
        help="Sliding window size for probability smoothing",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_webcam_pipeline(
        decision_seconds=args.decision_seconds,
        confidence_threshold=args.confidence_threshold,
        history_size=args.history_size,
        camera_index=args.camera_index,
        model_path=args.model_path,
        model_id=args.model_id,
    )

if __name__ == "__main__":
    main()
