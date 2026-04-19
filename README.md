# Trend Rewrite Backend

Lightweight backend for a hackathon MVP that:

- loads the 2026 YouTube trending CSV files from `archive/`
- builds a simple lexical similarity index
- returns similar trending videos for a user-provided title/description/tags
- produces structured rewrite suggestions from retrieved examples

This version intentionally avoids heavyweight dependencies so the team can move
fast even in a messy local Python environment.

## Setup

```bash
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

OpenAI model defaults live in `backend/app/config.py`, while secrets and
override values live in `.env`.

## Preprocessing

The first-pass retrieval dataset is built from a country CSV and written to
JSONL so it stays easy to inspect during the hackathon.

```bash
source .venv/bin/activate
python backend/preprocess_dataset.py --input archive/US_Trending.csv --output processed/US_Trending.processed.jsonl
```

You can also tune the phrase batch size for the embeddings API:

```bash
python backend/preprocess_dataset.py \
  --input archive/JP_Trending.csv \
  --output processed/JP_Trending.processed.jsonl \
  --embedding-batch-size 400
```

This preprocessing step:

- deduplicates by `video_id`
- keeps the highest-view row for each video
- preserves lightweight metadata for retrieval and LLM suggestions
- builds `topic / format / context / style` vectors from `title + tags`
