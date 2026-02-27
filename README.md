# ADS ScholarGraph

Production-style scaffold for a NASA ADS-powered knowledge graph and scientific literature recommendation system.

## What This Project Is

ADS ScholarGraph ingests scholarly metadata and citation links from NASA ADS, builds a Neo4j knowledge graph, and serves hybrid recommendations through an API and demo app. This repository currently implements data ingestion/normalization, citation expansion, Neo4j loading, and graph analytics feature write-back (Phases 0-3).

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

## Phase 2: Expand Citations & Load Neo4j

1. Expand citation edges from seed papers:
   ```bash
   python -m ads_scholargraph.pipeline.expand_citations \
     --seed data/processed/ads_star/papers.parquet \
     --mode references \
     --max-per-paper 200 \
     --out data/processed/ads_star/citations.parquet
   ```
   Recompute only when needed:
   ```bash
   python -m ads_scholargraph.pipeline.expand_citations \
     --seed data/processed/ads_star/papers.parquet \
     --mode references \
     --max-per-paper 200 \
     --out data/processed/ads_star/citations.parquet \
     --force
   ```

2. Start Neo4j (Docker Desktop must be running):
   ```bash
   docker compose up -d
   ```

3. Load processed tables into Neo4j:
   ```bash
   python -m ads_scholargraph.pipeline.load_neo4j \
     --indir data/processed/ads_star/ \
     --wipe
   ```

4. Open Neo4j Browser at http://localhost:7474 and run queries from `kg/queries.cypher`, for example:
   ```cypher
   MATCH (p:Paper)
   RETURN p.bibcode, p.title, p.citation_count
   ORDER BY p.citation_count DESC
   LIMIT 20;
   ```

## Phase 3: Graph Analytics

1. Ensure Neo4j is running and Phase 2 load is complete.

2. Run analytics in auto mode (try GDS first, then fall back to NetworkX):
   ```bash
   python -m ads_scholargraph.graph_analytics.run_analytics --mode auto
   ```

3. Optionally force a mode:
   ```bash
   python -m ads_scholargraph.graph_analytics.run_analytics --mode gds
   python -m ads_scholargraph.graph_analytics.run_analytics --mode networkx
   ```

4. Inspect analytics properties in Neo4j Browser using `kg/queries.cypher`, including:
- top papers by `pagerank`
- community size distribution by `community_id`
- top authors by `betweenness`
