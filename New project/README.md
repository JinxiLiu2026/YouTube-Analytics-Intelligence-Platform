# YouTube Insight Module

This module analyzes up to five public YouTube videos against a user-provided description.

It does four things:

1. fetches public transcript data with `youtube-transcript-api`
2. optionally fetches public stats and comments with the YouTube Data API
3. splits transcript text into 10-second chunks
4. uses an OpenAI model to predict the most engaging moments and generate content-quality suggestions

## What this module can and cannot do

- It can extract public transcripts when YouTube exposes them.
- It can run in transcript-only mode without any YouTube API key.
- It can optionally use public `viewCount`, `likeCount`, `commentCount`, and top comments if you later add a YouTube Data API key.
- It cannot access the true YouTube Studio audience retention chart for videos you do not own.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Config file

The module now looks for a root-level `config.json` by default. Put all model settings and API keys there:

```json
{
  "openai_api_key": "your_openai_api_key",
  "openai_model": "gpt-5-mini",
  "openai_embedding_model": "text-embedding-3-small",
  "openai_reasoning_effort": "low",
  "youtube_api_key": "",
  "analysis_mode": "fast",
  "transcript_languages": ["en", "en-US", "en-GB"],
  "chunk_seconds": 10,
  "analysis_window_chunks": 30,
  "candidate_moments_per_window": 3,
  "top_moments_per_video": 3,
  "max_comments_default": 40,
  "fast_chunk_seconds": 20,
  "fast_intro_seconds": 90,
  "fast_relevant_chunks": 18,
  "fast_numeric_chunks": 8,
  "max_parallel_video_analyses": 3
}
```

Create a local `config.json` from the example file and replace the placeholder values with your real keys:

```bash
cp config.example.json config.json
```

If your immediate goal is only to get transcripts and run LLM analysis, `openai_api_key` is the only required key.

## Environment variables

Create a `.env` file or export these variables:

```bash
OPENAI_API_KEY=...
YOUTUBE_API_KEY=
OPENAI_MODEL=gpt-5-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_REASONING_EFFORT=low
ANALYSIS_MODE=fast
```

`YOUTUBE_API_KEY` is optional. Without it, the module still extracts transcripts and runs analysis; it just skips public stats and comments.

Environment variables are still supported as a fallback, but `config.json` is the default source now.

## Fast Mode

`fast` mode is now the default. It speeds analysis up by:

- using larger transcript chunks
- always keeping the intro section
- keeping chunks that contain numbers/specs
- keeping the chunks most semantically related to the user description via embeddings
- sending a shorter transcript outline into the final LLM pass
- analyzing multiple videos in parallel

Use `mode="full"` if you want the slower whole-transcript pass.

## Run the API

```bash
uvicorn youtube_insights.api:app --reload
```

Request example:

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "description": "I want to make a YouTube video about AI tools for students.",
    "video_ids": ["abc123", "def456", "ghi789", "jkl012", "mno345"]
  }'
```

## Run the CLI

```bash
python3 -m youtube_insights.cli \
  --description "I want to make a YouTube video about AI tools for students." \
  --video-id abc123 \
  --video-id def456 \
  --video-id ghi789 \
  --video-id jkl012 \
  --video-id mno345
```

## Call it as a Python function

If your upstream system already has up to five `videoId`s, the simplest integration is a direct function call:

```python
from youtube_insights import analyze_youtube_videos

result = analyze_youtube_videos(
    description="I want to make a YouTube video about AI tools for students.",
    video_ids=["abc123", "def456", "ghi789", "jkl012", "mno345"],
    max_comments=30,
    mode="fast",
)

payload = result.model_dump(mode="json")
```

See [examples/invoke_module.py](/Users/jerry/Documents/New%20project/examples/invoke_module.py) for a runnable example.

## Output shape

The response includes:

- per-video metadata, stats, transcript availability, and comments
- additional `highlight_moments`
- cross-video patterns and practical creation suggestions

## Suggested integration point

Your upstream retrieval system can call this module after it has already selected up to five `videoId`s from your database.
