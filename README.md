# ADS ScholarGraph

Production-style scaffold for a NASA ADS-powered knowledge graph and scientific literature recommendation system.

## What This Project Is

ADS ScholarGraph ingests scholarly metadata and citation links from NASA ADS, builds a Neo4j knowledge graph, and serves hybrid recommendations through an API and demo app. This repository currently contains Phase 0 scaffolding only (project structure, config, local infra, and CI).

## Quickstart

1. Copy environment template and set credentials:
   ```bash
   cp .env.example .env
   ```
   Set `NEO4J_PASSWORD` (and `ADS_API_TOKEN` when available) in `.env`.
2. Start Neo4j:
   Ensure Docker Desktop is running first.
   ```bash
   docker compose up -d
   ```
3. Install project with development tooling:
   ```bash
   python -m pip install --upgrade pip
   pip install -e ".[dev]"
   ```
4. Run tests:
   ```bash
   pytest
   ```
5. Open Neo4j Browser: http://localhost:7474

Use username `neo4j` and the password from `.env`.

## Phase 1: Fetch & Normalize Data

1. Extract raw ADS records into JSON Lines (cached unless `--force`):
   ```bash
   python -m ads_scholargraph.pipeline.extract \
     --query "star" \
     --rows 200 \
     --max-results 2000 \
     --out data/raw/ads_star.jsonl
   ```

2. Refetch the same file when needed:
   ```bash
   python -m ads_scholargraph.pipeline.extract \
     --query "star" \
     --rows 200 \
     --max-results 2000 \
     --out data/raw/ads_star.jsonl \
     --force
   ```

3. Transform raw JSONL into normalized parquet tables:
   ```bash
   python -m ads_scholargraph.pipeline.transform \
     --in data/raw/ads_star.jsonl \
     --outdir data/processed/ads_star/
   ```

Expected outputs in `data/processed/ads_star/`:
- `papers.parquet`
- `authors.parquet`
- `paper_authors.parquet`
- `keywords.parquet`
- `paper_keywords.parquet`
- `venues.parquet`
- `paper_venues.parquet`
