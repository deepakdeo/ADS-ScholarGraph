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
