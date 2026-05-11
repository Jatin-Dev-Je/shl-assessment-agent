SHL Assessment Recommender

Conversational SHL assessment recommender built with FastAPI, LanceDB, Gemini embeddings, and Groq for the final response generation.

## What it does

- Scrapes the SHL assessment catalog into `catalog/catalog.json`
- Builds a LanceDB semantic index from the catalog
- Classifies user intent and retrieves relevant assessments
- Serves a `/chat` API for stateless conversations
- Includes an evaluation harness and unit tests

## Setup

1. Create a `.env` file from `.env.example`.
2. Install dependencies with `pip install -r requirements.txt`.
3. Ensure the catalog and index are available.

## Run

- API: `uvicorn api.main:app --host 0.0.0.0 --port 8000`
- Tests: `pytest -q`
- Scrape catalog: `python catalog/scraper.py`
- Build index: `python catalog/build_index.py`
- Evaluation: `python eval/harness.py`

## Notes

- The API is stateless and expects the full message history on every `/chat` request.
- Generated artifacts such as logs and debug HTML are ignored by git.
