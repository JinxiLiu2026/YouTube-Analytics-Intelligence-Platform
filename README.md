# YouTube Analytics Intelligence Platform

GitHub-ready hackathon demo for retrieval-grounded YouTube metadata improvement.

This repository includes a small runnable **US demo** version of the project.  
It does **not** include the full raw or full processed dataset.

## What is included in this demo

- frontend UI: `index.html`
- backend API: `backend/`
- matched-video insight module: `New project/`
- bundled processed demo index: `processed/US_Trending.processed.jsonl`
- bundled raw demo CSV: `demo_data/US_Trending_demo.csv`

The bundled demo data contains a small US subset only, so the GitHub version stays lightweight and runnable.

## What fields are used in the processed data

Each processed row keeps the lightweight fields needed by the demo:

- `video_id`
- `country`
- `title`
- `tags`
- `description`
- `views`
- `channel_title`
- `topic_vector`
- `format_vector`
- `context_vector`
- `style_vector`

## Full dataset

The full 2026 dataset should be downloaded from Kaggle:

[YouTube Trending Videos Dataset (2026)](https://www.kaggle.com/datasets/bsthere/youtube-trending-videos-stats-2026?resource=download)

After downloading, place the original CSV files in a local `archive/` folder at the project root.

Example:

```text
archive/
  US_Trending.csv
  JP_Trending.csv
  ...
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r "New project/requirements.txt"
cp config.example.json config.json
```

Then add your OpenAI API key to `config.json`.

## Run the GitHub demo

Start the backend:

```bash
source .venv/bin/activate
bash backend/run.sh
```

Then open `index.html` in your browser.

This GitHub demo is configured around the bundled US sample index:

- `processed/US_Trending.processed.jsonl`

## Rebuild the demo processed file

If you want to regenerate the bundled demo index from the included small CSV:

```bash
source .venv/bin/activate
python3 backend/preprocess_dataset.py \
  --input demo_data/US_Trending_demo.csv \
  --output processed/US_Trending.processed.jsonl
```

## Preprocess the full dataset

Once you have downloaded the full Kaggle CSVs into `archive/`, run:

```bash
source .venv/bin/activate
python3 backend/preprocess_dataset.py \
  --input archive/US_Trending.csv \
  --output processed/US_Trending.processed.jsonl
```

You can repeat the same command for other countries by changing the input and output file names.

Example:

```bash
python3 backend/preprocess_dataset.py \
  --input archive/JP_Trending.csv \
  --output processed/JP_Trending.processed.jsonl
```

## Notes

- Real API keys are not included in this repository.
- The GitHub version is intentionally trimmed to a small runnable demo.
- The full raw and processed datasets were removed because they are too large for GitHub.
