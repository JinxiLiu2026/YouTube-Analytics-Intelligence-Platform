from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from youtube_insights.config import Settings
from youtube_insights.facade import analyze_youtube_videos


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run youtube_insights analysis from a JSON payload file."
    )
    parser.add_argument(
        "--payload",
        type=Path,
        required=True,
        help="Path to a JSON payload containing description, video_ids, max_comments, and mode.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.json"),
        help="Path to the youtube_insights config.json file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.payload.read_text(encoding="utf-8"))
    settings = Settings.load(config_path=str(args.config))

    result = analyze_youtube_videos(
        description=payload["description"],
        video_ids=payload["video_ids"],
        max_comments=payload.get("max_comments"),
        settings=settings,
        mode=payload.get("mode"),
    )
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False))


if __name__ == "__main__":
    main()
