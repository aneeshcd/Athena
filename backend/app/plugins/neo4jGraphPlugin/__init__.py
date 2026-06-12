from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from app.plugins.neo4jGraphPlugin.graphRepository import Neo4jGraphRepository
    from app.plugins.neo4jGraphPlugin.types import (
        ImpactGraph,
        ImpactAnalysisResult,
        ImpactNoMatchResult,
        IngestSummary,
        OntologyRule,
        OntologyValidationResult,
        RequirementCandidate,
        SelectedRequirement,
    )

_repository: Neo4jGraphRepository | None = None


def get_repository() -> Neo4jGraphRepository:
    global _repository
    if _repository is None:
        from app.config import get_settings
        from app.plugins.neo4jGraphPlugin.graphRepository import Neo4jGraphRepository

        _repository = Neo4jGraphRepository(get_settings())
    return _repository


def close_repository() -> None:
    global _repository
    if _repository is not None:
        _repository.close()
        _repository = None


def ingestArtefact(filePath: str) -> IngestSummary:
    from app.plugins.neo4jGraphPlugin.ingestExcel import read_excel_artefact

    return get_repository().ingest(read_excel_artefact(filePath))


def clearGraph() -> None:
    get_repository().clear_graph()


def getOntology() -> list[OntologyRule]:
    return get_repository().get_ontology()


def searchRequirement(changeText: str) -> list[RequirementCandidate]:
    return get_repository().search_requirement(changeText)


def getImpactGraph(requirementId: str, depth: int = 2) -> ImpactGraph:
    return get_repository().get_impact_graph(requirementId, depth)


def analyzeRequirementChange(
    changeText: str,
    depth: int = 2,
    requestId: str | None = None,
    debug: bool = False,
) -> ImpactAnalysisResult | ImpactNoMatchResult:
    from app.plugins.neo4jGraphPlugin.graphRepository import _safe_depth
    from app.plugins.neo4jGraphPlugin.types import (
        ExcludedCandidate,
        ImpactAnalysisResult,
        ImpactDebug,
        ImpactGraph,
        ImpactNoMatchResult,
        SelectedRequirement,
    )

    current_request_id = requestId or str(uuid4())
    safe_depth = _safe_depth(depth)
    matches = searchRequirement(changeText)
    if not matches:
        debug_payload = ImpactDebug(
            traversalDepth=safe_depth,
            nodeCount=0,
            edgeCount=0,
        ) if debug else None
        return ImpactNoMatchResult(
            requestId=current_request_id,
            impactGraph=ImpactGraph(),
            message="No matching requirement found.",
            debug=debug_payload,
        )
    selected = matches[0]
    impact_graph = getImpactGraph(selected.id, safe_depth)
    debug_payload = None
    if debug:
        debug_payload = ImpactDebug(
            matchedRequirementId=selected.id,
            matchedRequirementName=selected.name,
            traversalDepth=safe_depth,
            nodeCount=len(impact_graph.nodes),
            edgeCount=len(impact_graph.edges),
            excludedCandidates=[
                ExcludedCandidate(id=match.id, reason="search candidate only, not part of traversal")
                for match in matches[1:]
            ],
        )
    return ImpactAnalysisResult(
        requestId=current_request_id,
        selectedRequirement=SelectedRequirement(
            id=selected.id,
            name=selected.name,
            description=selected.description,
            criticality=selected.criticality,
        ),
        impactGraph=impact_graph,
        debug=debug_payload,
    )


def validateEdgesAgainstOntology() -> OntologyValidationResult:
    return get_repository().validate_edges_against_ontology()


def generateImpactAnalysis(input):
    from app.plugins.neo4jGraphPlugin.impactAnalysisLLM import generateImpactAnalysis as _generate

    return _generate(input)
