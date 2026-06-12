from __future__ import annotations

import tempfile
import logging
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.config import get_settings
from app.plugins import neo4jGraphPlugin
from app.plugins.neo4jGraphPlugin.impactAnalysisLLM import (
    ImpactAnalysisFailed,
    ImpactAnalysisUnavailable,
    get_llm_health,
    generateImpactAnalysis,
    test_ollama_generation,
)
from app.plugins.neo4jGraphPlugin.types import ImpactGraph, LLMImpactAnalysisInput, SelectedRequirement


router = APIRouter(prefix="/api/graph", tags=["neo4j graph"])
logger = logging.getLogger(__name__)


class RequirementSearchRequest(BaseModel):
    changeText: str = Field(..., min_length=1)
    depth: int = Field(default=2, ge=1, le=3)
    requestId: str | None = None
    debug: bool = False


class ImpactRequest(BaseModel):
    requirementId: str = Field(..., min_length=1)
    depth: int = Field(default=2, ge=1, le=3)


class LLMImpactAnalysisRequest(BaseModel):
    changeText: str = Field(..., min_length=1)
    selectedRequirement: SelectedRequirement
    impactGraph: ImpactGraph
    debug: bool = False


class LLMTestRequest(BaseModel):
    prompt: str = Field(..., min_length=1)


@router.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith((".xlsx", ".xlsm")):
        raise HTTPException(status_code=400, detail="Upload an Excel .xlsx or .xlsm artefact.")
    suffix = Path(file.filename).suffix
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            handle.write(await file.read())
            temp_path = handle.name
        return neo4jGraphPlugin.ingestArtefact(temp_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Neo4j graph ingest failed: {exc}") from exc
    finally:
        if "temp_path" in locals():
            Path(temp_path).unlink(missing_ok=True)


@router.post("/requirement-search")
def requirement_search(request: RequirementSearchRequest):
    try:
        matches = neo4jGraphPlugin.searchRequirement(request.changeText)
        return [
            {
                "id": match.id,
                "name": match.name,
                "description": match.description,
                "criticality": match.criticality,
            }
            for match in matches
        ]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Requirement search failed: {exc}") from exc


@router.post("/impact-analysis")
def impact_analysis(request: LLMImpactAnalysisRequest):
    logger.info(
        "POST /api/graph/impact-analysis called: selected_requirement_present=%s node_count=%s edge_count=%s",
        request.selectedRequirement is not None,
        len(request.impactGraph.nodes),
        len(request.impactGraph.edges),
    )
    try:
        analysis = generateImpactAnalysis(LLMImpactAnalysisInput(**request.model_dump()))
        response = {"analysis": _public_analysis(analysis)}
        if request.debug:
            response["debug"] = _analysis_debug(analysis, request.impactGraph)
        return response
    except ImpactAnalysisUnavailable as exc:
        logger.exception("AI impact analysis unavailable: %s", exc)
        raise HTTPException(
            status_code=503,
            detail={
                "error": "AI_ANALYSIS_UNAVAILABLE",
                "message": "OpenAI API key is not configured",
                "details": str(exc),
            },
        ) from exc
    except ImpactAnalysisFailed as exc:
        logger.exception("AI impact analysis failed: %s", exc)
        raise HTTPException(
            status_code=502,
            detail={
                "error": "AI_ANALYSIS_FAILED",
                "message": "AI analysis could not be generated for this graph. Please review the impact map manually.",
            },
        ) from exc
    except ValueError as exc:
        logger.exception("AI impact analysis rejected: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected AI impact analysis error: %s", exc)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "AI_ANALYSIS_FAILED",
                "message": "AI analysis could not be generated for this graph. Please review the impact map manually.",
            },
        ) from exc


@router.post("/impact-from-change")
def impact_from_change(request: RequirementSearchRequest):
    try:
        return neo4jGraphPlugin.analyzeRequirementChange(
            request.changeText,
            request.depth,
            request.requestId,
            request.debug,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Impact analysis failed: {exc}") from exc


@router.post("/impact")
def impact(request: ImpactRequest):
    try:
        return neo4jGraphPlugin.getImpactGraph(request.requirementId, request.depth)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Impact traversal failed: {exc}") from exc


@router.get("/ontology")
def ontology():
    try:
        return neo4jGraphPlugin.getOntology()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ontology lookup failed: {exc}") from exc


@router.get("/health")
def health():
    try:
        return {"ok": neo4jGraphPlugin.get_repository().health()}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Neo4j connectivity failed: {exc}") from exc


@router.get("/llm-health")
def llm_health():
    return get_llm_health()


@router.post("/llm-test")
def llm_test(request: LLMTestRequest):
    return test_ollama_generation(request.prompt)


def _public_analysis(analysis):
    return {
        "summary": analysis.summary,
        "rippleEffects": [
            {
                "area": effect.area,
                "explanation": effect.explanation,
            }
            for effect in analysis.rippleEffects[:3]
        ],
        "suggestedNextSteps": analysis.suggestedNextSteps[:5],
    }


def _fallback_reason(analysis) -> str:
    for item in analysis.assumptionsAndLimitations:
        if item.startswith("Fallback reason: "):
            return item.replace("Fallback reason: ", "", 1)
    return ""


def _analysis_debug(analysis, impact_graph: ImpactGraph):
    settings = get_settings()
    nodes_sent = min(len(impact_graph.nodes), settings.ollama_max_context_nodes)
    edges_sent = min(len(impact_graph.edges), settings.ollama_max_context_edges)
    return {
        "provider": analysis.provider,
        "nodesSent": nodes_sent,
        "edgesSent": edges_sent,
        "wasTruncated": nodes_sent < len(impact_graph.nodes) or edges_sent < len(impact_graph.edges),
        "generationDurationMs": None,
        "retryUsed": None,
        "fallbackUsed": analysis.provider == "fallback",
        "safeError": _fallback_reason(analysis),
    }


@router.get("/validate")
def validate():
    try:
        return neo4jGraphPlugin.validateEdgesAgainstOntology()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ontology validation failed: {exc}") from exc
