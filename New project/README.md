# YouTube Insight Module

This folder contains the matched-video analysis module used by the main hackathon app.

It takes a small list of matched YouTube video IDs, samples transcript segments, and generates:

- highlight moments
- cross-video recurring patterns
- grounded creation tips

## How config works

This module reads the **root** `config.json` by default.

Start from:

- `../config.example.json`

Copy it to:

- `../config.json`

Then add your real OpenAI API key.

Environment variables are also supported through `.env`, but the root `config.json` is the default setup for this project.

## Notes

- `youtube_api_key` is optional
- transcript-based analysis can still run without it
- the main app already integrates this module, so most users do not need to run it separately
