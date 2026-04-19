from __future__ import annotations

import json

from youtube_insights import analyze_youtube_videos


def main() -> None:
    result = analyze_youtube_videos(
        description="I want to make a YouTube video about AI tools for students.",
        video_ids=[
            "abc123",
            "def456",
            "ghi789",
            "jkl012",
            "mno345",
        ],
        max_comments=30,
    )
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
