# Athena SE

Athena SE is a single-page systems engineering intelligence dashboard that turns requirement artifacts into a Neo4j-backed semantic knowledge graph, then uses GraphRAG-style traversals to analyze change impact.

## Architecture

- Frontend: React, Vite, Cytoscape.js
- Backend: FastAPI, Pydantic, Neo4j Python driver
- AI layer: OpenAI structured outputs for extraction, embeddings for semantic vectors, and graph-grounded summary synthesis
- Source of truth: Neo4j nodes and relationships, not the LLM

## Run Locally

```powershell
Copy-Item .env.example .env
docker compose up --build
```

Then open:

- Frontend: http://localhost:5173
- Backend docs: http://localhost:8000/docs
- Neo4j browser: http://localhost:7474

Neo4j login:

- Username: `neo4j`
- Password: `athena-password`

The app works without `OPENAI_API_KEY` by using a deterministic local extractor and graph-grounded summary fallback. Add an OpenAI key to enable LLM normalization, embeddings, and synthesized summaries.

## Pipeline

1. Ingestion receives PDF, Excel, CSV, text, or Markdown files.
2. A format profiler looks for likely requirement sections, sheet names, header aliases, IDs, owners, teams, verification columns, risk columns, and component allocations.
3. Normalization extracts requirements, components, interfaces, risks, tests, people, and teams into strict Pydantic models. With `OPENAI_API_KEY`, the LLM uses the profiler output to handle inconsistent engineer-authored layouts; without a key, deterministic table and text parsers still extract common formats.
4. Semantic processing embeds entity text, extracts relationships, and writes deterministic `MERGE` Cypher into Neo4j.
5. Change analysis maps an engineer's change description to the best graph node and traverses downstream paths.
6. Metrics aggregate impacted engineers, teams, effort, cost, delay, risk, safety, and AI confidence.
7. Output generation returns graph JSON, reasoning paths, source references, metrics, and a PDF-ready summary.

## Input Quality Handling

Athena SE is designed for mixed engineering document quality. It can inspect spreadsheet-style and document-style inputs with inconsistent names such as `Req ID`, `Requirement Number`, `Object Text`, `Description`, `Shall Statement`, `Owner`, `DRI`, `Verification`, `Test Case`, `Component`, `Allocated To`, `Risk`, and `Safety Critical`.

When an OpenAI API key is configured, the LLM is instructed to infer where the requirement content lives, ignore revision-history or decorative content, preserve source references, and lower confidence when extraction is ambiguous. The graph database remains the deterministic source of truth after extraction.

## Useful Endpoints

- `POST /api/ingest` uploads and normalizes an artifact.
- `POST /api/analyze` runs impact analysis for a natural-language change.
- `POST /api/report/pdf` generates a PDF report from the current dashboard state.
- `GET /api/graph` returns the current graph.
- `GET /health` checks backend and Neo4j connectivity.
