from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from openpyxl import Workbook

import pytest

from app.main import app
from app.plugins import neo4jGraphPlugin
from app.plugins.neo4jGraphPlugin.ingestExcel import read_excel_artefact
from app.plugins.neo4jGraphPlugin.ontologyValidator import sanitize_identifier, validate_edges
from app.plugins.neo4jGraphPlugin.types import (
    ImpactEdge,
    ImpactGraph,
    ImpactNode,
    RequirementCandidate,
)


def test_excel_ingestion_reads_required_sheets(tmp_path: Path):
    workbook_path = tmp_path / "artefact.xlsx"
    workbook = Workbook()
    nodes = workbook.active
    nodes.title = "nodes"
    nodes.append(["id", "type", "name", "description", "criticality"])
    nodes.append(["REQ-008", "Requirement", "Battery backup", "Battery backup duration", "High"])
    nodes.append(["SUB-001", "Subsystem", "Power subsystem", "UPS and battery pack", "Medium"])
    edges = workbook.create_sheet("edges")
    edges.append(["source_id", "target_id", "relationship", "description"])
    edges.append(["REQ-008", "SUB-001", "ALLOCATED_TO", "Allocated to power"])
    ontology = workbook.create_sheet("ontology")
    ontology.append(["source_entity", "relationship", "target_entity"])
    ontology.append(["Requirement", "ALLOCATED_TO", "Subsystem"])
    workbook.save(workbook_path)

    artefact = read_excel_artefact(str(workbook_path))

    assert artefact.nodes[0].id == "REQ-008"
    assert artefact.nodes[0].type == "Requirement"
    assert artefact.edges[0].relationship == "ALLOCATED_TO"
    assert artefact.ontology[0].target_entity == "Subsystem"


def test_ontology_validation_reports_invalid_edges(tmp_path: Path):
    artefact = read_excel_artefact(str(_sample_workbook(tmp_path)))

    invalid_edges = validate_edges(artefact.nodes, artefact.edges, artefact.ontology)

    assert [edge.relationship for edge in invalid_edges] == ["OWNED_BY"]
    assert invalid_edges[0].source_type == "Requirement"
    assert invalid_edges[0].target_type == "Team"


def test_requirement_search_contract_with_fake_repository():
    repository = FakeRepository()

    matches = repository.search_requirement("Adjust the backup power duration")

    assert matches[0].id == "REQ-008"
    assert matches[0].score > matches[1].score


def test_impact_traversal_contract_with_fake_repository():
    repository = FakeRepository()

    graph = repository.get_impact_graph("REQ-008", depth=2)

    assert {node.id: node.status for node in graph.nodes}["REQ-008"] == "selected"
    assert {"SUB-001", "TEST-004", "TEAM-POWER", "RISK-002"}.issubset({node.id for node in graph.nodes})
    assert all(edge.status == "ontology-link" for edge in graph.edges)


def test_change_analysis_returns_selected_requirement_and_impact_graph(monkeypatch):
    repository = FakeRepository()
    monkeypatch.setattr(neo4jGraphPlugin, "searchRequirement", repository.search_requirement)
    monkeypatch.setattr(neo4jGraphPlugin, "getImpactGraph", repository.get_impact_graph)

    result = neo4jGraphPlugin.analyzeRequirementChange("Adjust the backup power duration")

    assert result.selectedRequirement.id == "REQ-008"
    assert result.selectedRequirement.type == "Requirement"
    assert not hasattr(result.selectedRequirement, "score")
    assert {node.id: node.status for node in result.impactGraph.nodes}["REQ-008"] == "selected"
    assert result.impactGraph.edges


def test_requirement_search_endpoint_does_not_expose_scores(monkeypatch):
    repository = FakeRepository()
    monkeypatch.setattr(neo4jGraphPlugin, "searchRequirement", repository.search_requirement)
    client = TestClient(app)

    response = client.post("/api/graph/requirement-search", json={"changeText": "Adjust the backup power duration"})

    assert response.status_code == 200
    assert response.json()[0]["id"] == "REQ-008"
    assert "score" not in response.json()[0]


def test_required_graph_routes_are_registered():
    routes = {route.path for route in app.routes}

    assert {
        "/api/graph/ingest",
        "/api/graph/impact-analysis",
        "/api/graph/requirement-search",
        "/api/graph/impact",
        "/api/graph/ontology",
        "/api/graph/health",
    }.issubset(routes)


def test_dynamic_cypher_identifier_sanitization():
    assert sanitize_identifier("Allocated To") == "ALLOCATED_TO"
    assert sanitize_identifier("allocated-to") == "ALLOCATED_TO"
    with pytest.raises(ValueError):
        sanitize_identifier("Requirement`) DETACH DELETE n //")


class FakeRepository:
    def search_requirement(self, change_text: str):
        candidates = [
            RequirementCandidate(
                id="REQ-008",
                name="Battery backup duration",
                description="The battery backup shall last 45 minutes.",
                criticality="High",
                score=13.0,
            ),
            RequirementCandidate(
                id="REQ-002",
                name="Display brightness",
                description="The display shall be readable in sunlight.",
                criticality="Medium",
                score=1.0,
            ),
        ]
        return candidates

    def get_impact_graph(self, requirement_id: str, depth: int = 2):
        return ImpactGraph(
            nodes=[
                ImpactNode(
                    id=requirement_id,
                    label=requirement_id,
                    type="Requirement",
                    name="Battery backup duration",
                    status="selected",
                ),
                ImpactNode(id="SUB-001", label="SUB-001", type="Subsystem", name="Power", status="impacted"),
                ImpactNode(id="TEST-004", label="TEST-004", type="Test", name="Backup test", status="impacted"),
                ImpactNode(id="TEAM-POWER", label="TEAM-POWER", type="Team", name="Power team", status="impacted"),
                ImpactNode(id="RISK-002", label="RISK-002", type="Risk", name="Battery risk", status="impacted"),
            ],
            edges=[
                ImpactEdge(
                    id="REQ-008->ALLOCATED_TO->SUB-001",
                    source="REQ-008",
                    target="SUB-001",
                    relationship="ALLOCATED_TO",
                    status="ontology-link",
                ),
                ImpactEdge(
                    id="TEST-004->VALIDATES->REQ-008",
                    source="TEST-004",
                    target="REQ-008",
                    relationship="VALIDATES",
                    status="ontology-link",
                ),
            ],
        )


def _sample_workbook(tmp_path: Path) -> Path:
    workbook_path = tmp_path / "invalid-edge.xlsx"
    workbook = Workbook()
    nodes = workbook.active
    nodes.title = "nodes"
    nodes.append(["id", "type", "name", "description", "criticality"])
    nodes.append(["REQ-008", "Requirement", "Battery backup", "Battery backup duration", "High"])
    nodes.append(["SUB-001", "Subsystem", "Power subsystem", "UPS and battery pack", "Medium"])
    nodes.append(["TEAM-POWER", "Team", "Power team", "Owner team", "Low"])
    edges = workbook.create_sheet("edges")
    edges.append(["source_id", "target_id", "relationship", "description"])
    edges.append(["REQ-008", "SUB-001", "ALLOCATED_TO", "Allocated to power"])
    edges.append(["REQ-008", "TEAM-POWER", "OWNED_BY", "Invalid for this ontology"])
    ontology = workbook.create_sheet("ontology")
    ontology.append(["source_entity", "relationship", "target_entity"])
    ontology.append(["Requirement", "ALLOCATED_TO", "Subsystem"])
    workbook.save(workbook_path)
    return workbook_path
