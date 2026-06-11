from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.plugins import neo4jGraphPlugin


router = APIRouter(prefix="/api/graph", tags=["neo4j graph"])


class RequirementSearchRequest(BaseModel):
    changeText: str = Field(..., min_length=1)
    depth: int = Field(default=2, ge=1, le=5)


class ImpactRequest(BaseModel):
    requirementId: str = Field(..., min_length=1)
    depth: int = Field(default=2, ge=1, le=5)


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
def impact_analysis(request: RequirementSearchRequest):
    try:
        return neo4jGraphPlugin.analyzeRequirementChange(request.changeText, request.depth)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
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


@router.get("/validate")
def validate():
    try:
        return neo4jGraphPlugin.validateEdgesAgainstOntology()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ontology validation failed: {exc}") from exc
