from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from youtube_insights.config import Settings
from youtube_insights.schemas import AnalyzeVideosRequest
from youtube_insights.service import YouTubeInsightService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze public YouTube videos against a target description.")
    parser.add_argument("--description", help="Target video description or idea.")
    parser.add_argument("--description-file", help="Path to a text file containing the target description.")
    parser.add_argument("--video-id", action="append", dest="video_ids", required=True, help="YouTube video ID.")
    parser.add_argument("--max-comments", type=int, default=None, help="Maximum number of comments to pull per video.")
    parser.add_argument("--mode", choices=["fast", "full"], default=None, help="Analysis mode.")
    parser.add_argument("--output", help="Optional JSON output file path.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    description = _resolve_description(args.description, args.description_file)
    request = AnalyzeVideosRequest(
        description=description,
        video_ids=args.video_ids,
        max_comments=args.max_comments,
        mode=args.mode,
    )

    service = YouTubeInsightService(Settings.from_env())
    response = service.analyze(request)
    payload = response.model_dump(mode="json")
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)

    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        print(rendered)


def _resolve_description(description: Optional[str], description_file: Optional[str]) -> str:
    if description:
        return description.strip()
    if description_file:
        return Path(description_file).read_text(encoding="utf-8").strip()
    raise SystemExit("Either --description or --description-file is required.")


if __name__ == "__main__":
    main()
