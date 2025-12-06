"""Download and cache the HF model locally for offline/air-gapped use.

Usage:
  python scripts/cache_model.py --dest /models/fruits-and-vegetables-detector-36 \
    --model-id jazzmacedo/fruits-and-vegetables-detector-36

The destination directory is created if missing and can be baked into the Docker
image or mounted at runtime via MODEL_PATH.
"""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULT_MODEL_ID = "jazzmacedo/fruits-and-vegetables-detector-36"


def parse_args():
    parser = argparse.ArgumentParser(description="Cache HF model locally")
    parser.add_argument(
        "--dest",
        required=True,
        help="Destination directory for cached model files (will be created if missing)",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model id to download",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=args.model_id,
        local_dir=str(dest),
        local_dir_use_symlinks=False,
    )

    print(f"Cached model '{args.model_id}' to {dest}")


if __name__ == "__main__":
    main()
