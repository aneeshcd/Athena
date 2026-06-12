from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from app.ai import normalize_document
from app.config import DOCKER_DEFAULT_NEO4J_URI, Settings, get_settings, running_in_docker
from app.document_reader import extract_upload_text
from app.graph import GraphRepository, MemoryGraphRepository, create_repository
from app.impact import analyze_change
from app.models import GraphEdge, GraphNode, GraphPayload, ImpactAnalysisRequest, IngestionResponse, PdfReportRequest
from app.plugins.neo4jGraphPlugin import close_repository as close_neo4j_graph_plugin
from app.plugins.neo4jGraphPlugin.routes import router as neo4j_graph_router
from app.report import build_pdf_report


repository: GraphRepository | MemoryGraphRepository | None = None
logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global repository
    settings = get_settings()
    logger.info("[LLM] provider=%s", settings.llm_provider or "ollama")
    logger.info("[LLM] ollamaBaseUrl=%s", settings.ollama_base_url)
    logger.info("[LLM] ollamaModel=%s", settings.ollama_model)
    logger.info("[LLM] ollamaTimeoutMs=%s", settings.ollama_timeout_ms)
    logger.info("[LLM] ollamaMaxContextNodes=%s", settings.ollama_max_context_nodes)
    logger.info("[LLM] ollamaMaxContextEdges=%s", settings.ollama_max_context_edges)
    logger.info("[LLM] openaiConfigured=%s", bool(settings.openai_api_key))
    logger.info("[Config] os.environ NEO4J_URI=%s", os.environ.get("NEO4J_URI", "<unset>"))
    logger.info("[Config] Settings.neo4j_uri=%s", settings.neo4j_uri)
    if running_in_docker() and "NEO4J_URI" not in os.environ and settings.neo4j_uri != DOCKER_DEFAULT_NEO4J_URI:
        raise RuntimeError(f"Docker backend default NEO4J_URI must be {DOCKER_DEFAULT_NEO4J_URI}.")
    logger.info("[Neo4j] Connecting to %s", settings.neo4j_uri)
    repository = create_repository(settings)
    repository.ensure_constraints()
    yield
    close_neo4j_graph_plugin()
    if repository:
        repository.close()


app = FastAPI(title="Athena SE API", version="0.1.0", lifespan=lifespan)
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(neo4j_graph_router)


def get_repository() -> GraphRepository | MemoryGraphRepository:
    if repository is None:
        raise HTTPException(status_code=503, detail="Graph repository is not initialized.")
    return repository


@app.get("/health")
def health(repo: GraphRepository | MemoryGraphRepository = Depends(get_repository)):
    return {"ok": True, "graph": repo.health(), "repository": repo.__class__.__name__}


@app.post("/api/ingest", response_model=IngestionResponse)
async def ingest(
    file: UploadFile = File(...),
    repo: GraphRepository | MemoryGraphRepository = Depends(get_repository),
    current_settings: Settings = Depends(get_settings),
):
    raw_text = await extract_upload_text(file)
    if not raw_text.strip():
        raise HTTPException(status_code=400, detail="No readable text was found in the uploaded artifact.")

    artifact = normalize_document(raw_text, file.filename or "uploaded-artifact", current_settings)
    repo.clear_graph()
    cypher_statements = repo.ingest_artifact(artifact)
    graph = graph_from_artifact(artifact)
    return IngestionResponse(artifact=artifact, graph=graph, cypher_statements=cypher_statements)


@app.get("/api/graph")
def graph(repo: GraphRepository | MemoryGraphRepository = Depends(get_repository)):
    return repo.get_graph()


@app.post("/api/analyze")
def analyze(
    request: ImpactAnalysisRequest,
    repo: GraphRepository | MemoryGraphRepository = Depends(get_repository),
    current_settings: Settings = Depends(get_settings),
):
    try:
        if not repo.get_graph(limit=1).nodes:
            raise ValueError("No active graph is loaded. Upload and ingest a requirement artifact first.")
        return analyze_change(request.change_request, repo, current_settings)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/report/pdf")
def pdf_report(request: PdfReportRequest):
    pdf = build_pdf_report(request.analysis)
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="athena-se-impact-report.pdf"'},
    )


def graph_from_artifact(artifact) -> GraphPayload:
    nodes = [
        GraphNode(
            id=entity.id,
            label=entity.title,
            type=entity.type,
            source_ref=entity.source_ref,
            owner=entity.owner,
            team=entity.team,
            effort_hours=entity.effort_hours,
            cost_rate=entity.cost_rate,
            delay_days=entity.delay_days,
            safety_critical=entity.safety_critical,
            confidence=entity.confidence,
        )
        for entity in artifact.entities
    ]
    node_ids = {node.id for node in nodes}
    edges = [
        GraphEdge(
            id=f"{relationship.source_id}->{relationship.type}->{relationship.target_id}",
            source=relationship.source_id,
            target=relationship.target_id,
            type=relationship.type,
            rationale=relationship.rationale,
            confidence=relationship.confidence,
        )
        for relationship in artifact.relationships
        if relationship.source_id in node_ids and relationship.target_id in node_ids
    ]
    return GraphPayload(nodes=nodes, edges=edges)
