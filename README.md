# Athena

Athena is a minimal Neo4j knowledge graph prototype for requirement change management.

The current prototype ingests an Excel artefact with three sheets:

- `nodes`: `id`, `type`, `name`, `description`, `criticality`
- `edges`: `source_id`, `target_id`, `relationship`, `description`
- `ontology`: `source_entity`, `relationship`, `target_entity`

It stores the graph in Neo4j, validates relationships against the declared ontology, identifies the best starting `Requirement` from engineer-entered change text, and traverses ontology-approved relationships to show impacted connected nodes.

The prototype intentionally does not implement cost calculation, delay calculation, man-hour calculation, LLM summaries, vector embeddings, Text2Cypher, agentic RAG, or authentication changes.

## Architecture

- Frontend: React, Vite, Cytoscape.js
- Backend: FastAPI, Pydantic, Neo4j Python driver
- Neo4j plugin module: `backend/app/plugins/neo4jGraphPlugin`
- Source of truth: Neo4j `GraphNode` and `OntologyRule` data

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

The compose file keeps the local frontend and backend host ports at `5173` and `8000`.

## Environment

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=athena-password
NEO4J_DATABASE=neo4j
```

`NEO4J_DATABASE` is optional and defaults to `neo4j`.

## Neo4j Graph Plugin

The plugin exposes:

- `ingestArtefact(filePath: str)`
- `clearGraph()`
- `getOntology()`
- `searchRequirement(changeText: str)`
- `analyzeRequirementChange(changeText: str, depth: int = 2)`
- `getImpactGraph(requirementId: str, depth: int = 2)`
- `validateEdgesAgainstOntology()`

Dynamic labels and relationship types are sanitized before being inserted into Cypher. Spaces and hyphens become underscores, and unsafe identifiers are rejected.

## API Endpoints

- `POST /api/graph/ingest`: upload an Excel artefact and build the graph.
- `POST /api/graph/impact-analysis`: identify the starting `Requirement` internally and return the impact map.
- `POST /api/graph/requirement-search`: search matching `Requirement` nodes without exposing scores.
- `POST /api/graph/impact`: traverse impacted nodes from a selected requirement.
- `GET /api/graph/ontology`: return ontology rules.
- `GET /api/graph/health`: verify Neo4j connectivity.

Impact-analysis body:

```json
{
  "changeText": "<engineer-entered requirement change text>"
}
```

The backend uses full-text search internally to choose the most likely starting requirement, then returns the selected requirement and the existing impact graph. The frontend shows the impact map and impacted nodes/relationships instead of search rankings or relevance scores.

## Tests

From the backend folder:

```powershell
pytest
```

The basic tests cover Excel ingestion, ontology validation, the requirement-search contract, and the impact-traversal response contract.
