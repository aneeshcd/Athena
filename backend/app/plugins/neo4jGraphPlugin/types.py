from __future__ import annotations

from pydantic import BaseModel, Field


class ArtefactNode(BaseModel):
    id: str
    type: str
    name: str
    description: str = ""
    criticality: str = ""


class ArtefactEdge(BaseModel):
    source_id: str
    target_id: str
    relationship: str
    description: str = ""


class OntologyRule(BaseModel):
    source_entity: str
    relationship: str
    target_entity: str


class InvalidEdge(BaseModel):
    source_id: str
    target_id: str
    relationship: str
    source_type: str | None = None
    target_type: str | None = None
    reason: str


class OntologyValidationResult(BaseModel):
    valid: bool
    invalid_edges: list[InvalidEdge] = Field(default_factory=list)


class IngestSummary(BaseModel):
    nodes_created: int
    edges_created: int
    ontology_rules_created: int
    invalid_edges: list[InvalidEdge] = Field(default_factory=list)


class RequirementCandidate(BaseModel):
    id: str
    name: str
    description: str = ""
    criticality: str = ""
    score: float


class SelectedRequirement(BaseModel):
    id: str
    type: str = "Requirement"
    name: str
    description: str = ""
    criticality: str = ""


class ImpactNode(BaseModel):
    id: str
    label: str
    type: str
    name: str
    description: str = ""
    criticality: str = ""
    status: str


class ImpactEdge(BaseModel):
    id: str
    source: str
    target: str
    relationship: str
    description: str = ""
    status: str


class ImpactGraph(BaseModel):
    nodes: list[ImpactNode] = Field(default_factory=list)
    edges: list[ImpactEdge] = Field(default_factory=list)


class ImpactAnalysisResult(BaseModel):
    selectedRequirement: SelectedRequirement
    impactGraph: ImpactGraph


class ParsedArtefact(BaseModel):
    nodes: list[ArtefactNode]
    edges: list[ArtefactEdge]
    ontology: list[OntologyRule]
