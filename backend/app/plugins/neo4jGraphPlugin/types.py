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
    hop: int | None = None


class ImpactEdge(BaseModel):
    id: str
    source: str
    target: str
    relationship: str
    description: str = ""
    status: str
    hop: int | None = None


class ImpactGraph(BaseModel):
    nodes: list[ImpactNode] = Field(default_factory=list)
    edges: list[ImpactEdge] = Field(default_factory=list)


class ExcludedCandidate(BaseModel):
    id: str
    reason: str


class ImpactDebug(BaseModel):
    matchedRequirementId: str | None = None
    matchedRequirementName: str | None = None
    traversalDepth: int
    nodeCount: int
    edgeCount: int
    cypherQueryName: str = "impactTraversalFromRequirement"
    excludedCandidates: list[ExcludedCandidate] = Field(default_factory=list)


class ImpactAnalysisResult(BaseModel):
    requestId: str
    selectedRequirement: SelectedRequirement
    impactGraph: ImpactGraph
    debug: ImpactDebug | None = None


class ImpactNoMatchResult(BaseModel):
    requestId: str
    selectedRequirement: None = None
    impactGraph: ImpactGraph = Field(default_factory=ImpactGraph)
    message: str
    debug: ImpactDebug | None = None


class LLMImpactAnalysisInput(BaseModel):
    changeText: str
    selectedRequirement: SelectedRequirement
    impactGraph: ImpactGraph


class RippleEffect(BaseModel):
    area: str
    explanation: str
    affectedNodes: list[str] = Field(default_factory=list)


class AffectedNodeSummary(BaseModel):
    nodeId: str
    nodeName: str
    nodeType: str
    whyItMatters: str


class LLMImpactAnalysisResult(BaseModel):
    provider: str = "fallback"
    summary: str
    rippleEffects: list[RippleEffect] = Field(default_factory=list)
    affectedNodeSummary: list[AffectedNodeSummary] = Field(default_factory=list)
    suggestedNextSteps: list[str] = Field(default_factory=list)
    engineeringReviewChecklist: list[str] = Field(default_factory=list)
    assumptionsAndLimitations: list[str] = Field(default_factory=list)
    humanInTheLoopNotice: str


class LLMImpactAnalysisResponse(BaseModel):
    analysis: LLMImpactAnalysisResult


class ParsedArtefact(BaseModel):
    nodes: list[ArtefactNode]
    edges: list[ArtefactEdge]
    ontology: list[OntologyRule]
