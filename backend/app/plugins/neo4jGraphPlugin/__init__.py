from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.plugins.neo4jGraphPlugin.graphRepository import Neo4jGraphRepository
    from app.plugins.neo4jGraphPlugin.types import (
        ImpactGraph,
        ImpactAnalysisResult,
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


def analyzeRequirementChange(changeText: str, depth: int = 2) -> ImpactAnalysisResult:
    from app.plugins.neo4jGraphPlugin.types import ImpactAnalysisResult, SelectedRequirement

    matches = searchRequirement(changeText)
    if not matches:
        raise ValueError("No matching Requirement node was found for the entered change.")
    selected = matches[0]
    return ImpactAnalysisResult(
        selectedRequirement=SelectedRequirement(
            id=selected.id,
            name=selected.name,
            description=selected.description,
            criticality=selected.criticality,
        ),
        impactGraph=getImpactGraph(selected.id, depth),
    )


def validateEdgesAgainstOntology() -> OntologyValidationResult:
    return get_repository().validate_edges_against_ontology()
