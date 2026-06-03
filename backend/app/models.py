from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from pydantic import BaseModel, Field


EntityType = Literal[
    "REQUIREMENT",
    "COMPONENT",
    "INTERFACE",
    "RISK",
    "TEST",
    "ENGINEER",
    "TEAM",
    "DOCUMENT",
]

RelationshipType = Literal[
    "VALIDATES",
    "DEPENDS_ON",
    "CONFLICTS_WITH",
    "DERIVED_FROM",
    "IMPLEMENTS",
    "OWNED_BY",
    "BELONGS_TO",
    "MITIGATES",
    "AFFECTS",
    "SEMANTICALLY_SIMILAR",
]


class EngineeringEntity(BaseModel):
    id: str = Field(..., description="Stable artifact or semantic identifier, e.g. REQ-BRAKE-001.")
    type: EntityType
    title: str
    description: str
    source_ref: str = Field(..., description="Traceable source reference such as filename:page:line.")
    author: str | None = None
    owner: str | None = None
    team: str | None = None
    artifact_id: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    effort_hours: float = Field(default=8.0, ge=0)
    cost_rate: float = Field(default=95.0, ge=0)
    delay_days: float = Field(default=0.5, ge=0)
    safety_critical: bool = False
    confidence: float = Field(default=0.75, ge=0, le=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EngineeringRelationship(BaseModel):
    source_id: str
    target_id: str
    type: RelationshipType
    rationale: str
    confidence: float = Field(default=0.75, ge=0, le=1)


class NormalizedArtifact(BaseModel):
    document_id: str
    filename: str
    extracted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    entities: list[EngineeringEntity]
    relationships: list[EngineeringRelationship]
    warnings: list[str] = Field(default_factory=list)


class GraphNode(BaseModel):
    id: str
    label: str
    type: EntityType
    source_ref: str | None = None
    owner: str | None = None
    team: str | None = None
    effort_hours: float = 8.0
    cost_rate: float = 95.0
    delay_days: float = 0.5
    safety_critical: bool = False
    confidence: float = 0.75


class GraphEdge(BaseModel):
    id: str
    source: str
    target: str
    type: RelationshipType
    rationale: str = ""
    confidence: float = 0.75


class GraphPayload(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class MetricPayload(BaseModel):
    required_man_hours: float
    cost_impact: float
    engineers_affected: int
    teams_affected: int
    project_delay_days: float
    risk_category: Literal["Low", "Medium", "High", "Critical"]
    safety_impact: Literal["None", "Low", "Moderate", "High"]
    ai_confidence_level: float


class ImpactFinding(BaseModel):
    title: str
    severity: Literal["info", "warning", "critical"]
    evidence: list[str]


class ImpactAnalysisRequest(BaseModel):
    change_request: str = Field(..., min_length=3)


class ImpactAnalysisResponse(BaseModel):
    changed_node_id: str
    changed_node_label: str
    graph: GraphPayload
    summary: str
    reasoning_paths: list[str]
    source_references: list[str]
    confidence_score: float
    metrics: MetricPayload
    findings: list[ImpactFinding]
    next_steps: list[str]


class IngestionResponse(BaseModel):
    artifact: NormalizedArtifact
    graph: GraphPayload
    cypher_statements: list[str]


class PdfReportRequest(BaseModel):
    analysis: ImpactAnalysisResponse
