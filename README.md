# Athena

Athena is a minimal Neo4j knowledge graph prototype for requirement change management.

The current prototype ingests an Excel artefact with three sheets:

- `nodes`: `id`, `type`, `name`, `description`, `criticality`
- `edges`: `source_id`, `target_id`, `relationship`, `description`
- `ontology`: `source_entity`, `relationship`, `target_entity`

It stores the graph in Neo4j, validates relationships against the declared ontology, identifies the best starting `Requirement` from engineer-entered change text, and traverses ontology-approved relationships to show impacted connected nodes.

The prototype intentionally does not implement cost calculation, delay calculation, man-hour calculation, vector embeddings, Text2Cypher, agentic RAG, or authentication changes.

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
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_MODEL=llama3.2:3b
OLLAMA_TIMEOUT_MS=300000
OLLAMA_MAX_CONTEXT_NODES=20
OLLAMA_MAX_CONTEXT_EDGES=30
OLLAMA_MAX_DESCRIPTION_CHARS=180
OLLAMA_WARMUP_ON_START=true
# OPENAI_API_KEY=
# OPENAI_MODEL=gpt-4.1-mini
```

`NEO4J_DATABASE` is optional and defaults to `neo4j`.
`LLM_PROVIDER` is optional and defaults to `ollama`. Supported values are `ollama`, `auto`, `openai`, and `fallback`.
`OPENAI_MODEL` is optional and defaults to `gpt-4.1-mini`.
`OLLAMA_MODEL` is optional and defaults to `llama3.2:3b`.
`OLLAMA_TIMEOUT_MS` defaults to `300000` so local models have up to 300 seconds to load and generate.
`OLLAMA_MAX_CONTEXT_NODES` and `OLLAMA_MAX_CONTEXT_EDGES` limit the graph context sent to Ollama.
`OLLAMA_MAX_DESCRIPTION_CHARS` limits selected-requirement description text sent to the local LLM.

## Using Ollama for Free Local AI Analysis

Install Ollama locally.

Linux:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Windows/macOS:

Install Ollama from https://ollama.com.

Pull the recommended small prototype model:

```bash
ollama pull llama3.2:3b
```

Alternative:

```bash
ollama pull qwen2.5:3b
```

Manual model test:

```bash
ollama run llama3.2:3b
```

Check the local Ollama API:

```bash
curl http://localhost:11434/api/tags
```

For Docker Desktop, set the backend to reach host Ollama through `host.docker.internal`:

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_MODEL=llama3.2:3b
```

Then restart:

```bash
sudo docker-compose down
sudo docker-compose up --build -d
sudo docker-compose logs -f backend
```

Provider behavior:

- `LLM_PROVIDER=ollama`: use Ollama; if unavailable, use rule-based fallback.
- `LLM_PROVIDER=auto`: try OpenAI, then Ollama, then rule-based fallback.
- `LLM_PROVIDER=openai`: use OpenAI only.
- `LLM_PROVIDER=fallback`: use deterministic rule-based analysis only.

The AI analysis endpoint never exposes API keys to the frontend. OpenAI is optional and is not used unless `LLM_PROVIDER=openai` or `LLM_PROVIDER=auto` is explicitly configured. If OpenAI fails with quota errors such as `429 insufficient_quota`, `auto` falls through to Ollama and then to the rule-based fallback if needed.

Manual Docker validation:

```bash
OLLAMA_HOST=0.0.0.0:11434 ollama serve
ollama pull llama3.2:3b
curl http://localhost:11434/api/tags
sudo docker-compose down
sudo docker-compose up --build -d
sudo docker-compose exec backend printenv LLM_PROVIDER
sudo docker-compose exec backend printenv OLLAMA_BASE_URL
sudo docker-compose exec backend printenv OLLAMA_MODEL
sudo docker-compose exec backend python3 -c "import urllib.request; print(urllib.request.urlopen('http://host.docker.internal:11434/api/tags', timeout=10).read().decode())"
curl http://localhost:8000/api/graph/llm-health
curl -X POST http://localhost:8000/api/graph/llm-test -H "Content-Type: application/json" -d '{"prompt":"Return JSON only with summary saying hello."}'
```

Tiny generation test from the backend container:

```bash
sudo docker-compose exec backend python3 - <<'PY'
import urllib.request, json, time
payload = {
  "model": "llama3.2:3b",
  "messages": [{"role": "user", "content": "Return JSON only: {\"summary\":\"hello\"}"}],
  "stream": False,
  "format": "json"
}
data = json.dumps(payload).encode()
req = urllib.request.Request(
  "http://host.docker.internal:11434/api/chat",
  data=data,
  headers={"Content-Type": "application/json"},
  method="POST"
)
start = time.time()
with urllib.request.urlopen(req, timeout=180) as r:
  print("duration", time.time() - start)
  print(r.read().decode())
PY
```

If `llama3.2:3b` is still slow on your machine, use a faster local model:

```bash
ollama pull llama3.2:1b
# then set OLLAMA_MODEL=llama3.2:1b
```

or:

```bash
ollama pull qwen2.5:3b
# then set OLLAMA_MODEL=qwen2.5:3b
```

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
- `POST /api/graph/impact-from-change`: identify one starting `Requirement` internally and return a fresh impact map for that request.
- `POST /api/graph/impact-analysis`: generate advisory AI impact analysis from an already-computed impact graph.
- `POST /api/graph/requirement-search`: search matching `Requirement` nodes without exposing scores.
- `POST /api/graph/impact`: traverse impacted nodes from a selected requirement.
- `GET /api/graph/ontology`: return ontology rules.
- `GET /api/graph/health`: verify Neo4j connectivity.
- `GET /api/graph/llm-health`: verify active LLM provider config and Ollama model reachability.
- `POST /api/graph/llm-test`: run a tiny Ollama generation test for debugging local LLM connectivity.

Impact-analysis body:

```json
{
  "changeText": "<engineer-entered requirement change text>",
  "depth": 2,
  "requestId": "<frontend-generated uuid>",
  "debug": false
}
```

The backend uses full-text search internally to choose exactly one starting requirement, then returns the selected requirement and the existing ontology traversal result. The frontend clears the previous graph before every new request, ignores late responses whose `requestId` is no longer current, and shows the latest impact map only.

LLM analysis request body:

```json
{
  "changeText": "<engineer-entered requirement change text>",
  "selectedRequirement": {
    "id": "REQ-008",
    "type": "Requirement",
    "name": "Battery Backup Duration",
    "description": "...",
    "criticality": "Critical"
  },
  "impactGraph": {
    "nodes": [],
    "edges": []
  }
}
```

The LLM endpoint never queries Neo4j and never decides which nodes are impacted. It receives only the current change text, selected requirement, and already-computed impact graph. The normal frontend response includes only `summary`, `rippleEffects`, and `suggestedNextSteps`; provider and diagnostic details are returned only when `debug=true`.

Debug mode is available for developers by setting `"debug": true`. Debug responses include the matched requirement, traversal depth, node and edge counts, the query name, and excluded search-only candidates. Debug details are not shown in the normal frontend.

Traversal rules:

- Start from the selected `Requirement` only.
- Traverse only relationships declared in `OntologyRule`.
- Preserve actual Neo4j relationship direction in returned edges.
- Clamp depth to `1`, `2`, or `3`; default is `2`.
- Deduplicate nodes by `node.id` and edges by `source::relationship::target`.
- Include `hop` values where available so direct and ripple impacts can be distinguished.

## Tests

From the backend folder:

```powershell
pytest
```

The backend tests cover Excel ingestion, ontology validation, deterministic selected-requirement handling, impact traversal contracts, no-match empty graph behavior, debug output, duplicate prevention, supported depth clamping, and score-free search responses.

The frontend behavior test checks that new submissions clear the old graph, request IDs are sent and tracked, late responses are ignored, the graph remounts for fresh results, no search scores/rankings are rendered, and no-match results show a clear message.
It also checks that AI impact analysis clears on new submissions, displays a loading state, calls only backend LLM endpoints, handles provider fallback without breaking the graph, ignores stale LLM responses, and shows the human-in-the-loop notice.
