from __future__ import annotations

from app.ai import synthesize_summary
from app.config import Settings
from app.graph import GraphRepository, MemoryGraphRepository
from app.models import (
    GraphNode,
    ImpactAnalysisResponse,
    ImpactFinding,
    MetricPayload,
)


def analyze_change(
    change_request: str,
    repository: GraphRepository | MemoryGraphRepository,
    settings: Settings,
) -> ImpactAnalysisResponse:
    changed_node = repository.match_change_node(change_request)
    if changed_node is None:
        raise ValueError("No graph nodes are available. Upload a requirement artifact first.")

    graph, raw_paths = repository.downstream_impact(changed_node.id)
    if not graph.nodes:
        graph.nodes = [changed_node]

    metrics = _calculate_metrics(graph.nodes)
    findings = _detect_conflicts(repository, graph.nodes)
    metrics = _adjust_risk(metrics, findings, graph.nodes)
    reasoning_paths = _format_paths(raw_paths, graph.nodes)
    source_references = sorted({node.source_ref for node in graph.nodes if node.source_ref})
    confidence_score = _confidence(graph.nodes)
    metrics.ai_confidence_level = confidence_score
    next_steps = _next_steps(metrics, findings)

    context = {
        "nodes": [node.model_dump() for node in graph.nodes],
        "edges": [edge.model_dump() for edge in graph.edges],
    }
    summary = synthesize_summary(
        change_request=change_request,
        changed_node=changed_node,
        graph_context=context,
        metrics=metrics,
        reasoning_paths=reasoning_paths,
        source_references=source_references,
        findings=[finding.model_dump() for finding in findings],
        settings=settings,
    )

    return ImpactAnalysisResponse(
        changed_node_id=changed_node.id,
        changed_node_label=changed_node.label,
        graph=graph,
        summary=summary,
        reasoning_paths=reasoning_paths,
        source_references=source_references,
        confidence_score=confidence_score,
        metrics=metrics,
        findings=findings,
        next_steps=next_steps,
    )


def _calculate_metrics(nodes: list[GraphNode]) -> MetricPayload:
    engineers = {node.owner for node in nodes if node.owner}
    teams = {node.team for node in nodes if node.team}
    required_hours = 0.0
    delay_days = 0.0
    cost = 0.0

    type_weights = {
        "REQUIREMENT": (12, 1.0),
        "COMPONENT": (24, 2.0),
        "INTERFACE": (18, 1.5),
        "RISK": (16, 1.5),
        "TEST": (10, 0.75),
        "DOCUMENT": (2, 0.1),
        "ENGINEER": (0, 0),
        "TEAM": (0, 0),
    }

    for node in nodes:
        hours, delay = type_weights.get(node.type, (8, 0.5))
        if node.safety_critical:
            hours *= 1.6
            delay *= 1.8
        required_hours += hours
        delay_days += delay
        cost += hours * (140 if node.safety_critical else 125)

    risk = "Low"
    if len(nodes) >= 12 or any(node.safety_critical for node in nodes):
        risk = "High"
    elif len(nodes) >= 6:
        risk = "Medium"

    safety = "None"
    safety_count = sum(1 for node in nodes if node.safety_critical)
    if safety_count >= 3:
        safety = "High"
    elif safety_count >= 1:
        safety = "Moderate"

    return MetricPayload(
        required_man_hours=round(required_hours, 1),
        cost_impact=round(cost, 2),
        engineers_affected=len(engineers),
        teams_affected=len(teams),
        project_delay_days=round(delay_days, 1),
        risk_category=risk,  # type: ignore[arg-type]
        safety_impact=safety,  # type: ignore[arg-type]
        ai_confidence_level=0.75,
    )


def _detect_conflicts(
    repository: GraphRepository | MemoryGraphRepository,
    nodes: list[GraphNode],
) -> list[ImpactFinding]:
    node_ids = [node.id for node in nodes]
    findings: list[ImpactFinding] = []
    orphan_ids = repository.orphan_requirements(node_ids)
    if orphan_ids:
        findings.append(
            ImpactFinding(
                title="Missing verification coverage",
                severity="critical",
                evidence=[f"{node_id} has no incoming VALIDATES relationship" for node_id in orphan_ids],
            )
        )

    safety_nodes = [node.id for node in nodes if node.safety_critical]
    if safety_nodes:
        findings.append(
            ImpactFinding(
                title="Safety-critical propagation detected",
                severity="warning",
                evidence=[f"{node_id} is marked safety critical" for node_id in safety_nodes],
            )
        )

    if not findings:
        findings.append(
            ImpactFinding(
                title="No structural conflicts detected",
                severity="info",
                evidence=["Traversal did not expose orphan requirements or safety-critical contradictions."],
            )
        )

    return findings


def _adjust_risk(metrics: MetricPayload, findings: list[ImpactFinding], nodes: list[GraphNode]) -> MetricPayload:
    if any(finding.severity == "critical" for finding in findings):
        metrics.risk_category = "Critical"
    elif any(finding.severity == "warning" for finding in findings) and metrics.risk_category == "Low":
        metrics.risk_category = "Medium"

    if any(node.safety_critical for node in nodes) and metrics.safety_impact == "None":
        metrics.safety_impact = "Moderate"

    return metrics


def _confidence(nodes: list[GraphNode]) -> float:
    if not nodes:
        return 0.0
    return round(sum(node.confidence for node in nodes) / len(nodes), 2)


def _format_paths(paths: list[list[str]], nodes: list[GraphNode]) -> list[str]:
    labels = {node.id: node.label for node in nodes}
    formatted = []
    for path in paths[:12]:
        formatted.append(" -> ".join(f"{labels.get(node_id, node_id)} ({node_id})" for node_id in path))
    return formatted or ["No downstream traversal path was found beyond the changed node."]


def _next_steps(metrics: MetricPayload, findings: list[ImpactFinding]) -> list[str]:
    steps = [
        "Assign an engineer to validate the changed requirement mapping before implementation.",
        "Review every impacted component and test owner listed in the graph.",
        "Update verification artifacts for each downstream requirement before release approval.",
    ]
    if metrics.risk_category in {"High", "Critical"}:
        steps.insert(1, "Schedule a cross-functional impact review with systems, safety, verification, and program leads.")
    if any(finding.severity == "critical" for finding in findings):
        steps.append("Create remediation tasks for missing validation links before accepting the change.")
    return steps
