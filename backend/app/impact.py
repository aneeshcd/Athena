from __future__ import annotations

import math

from app.ai import synthesize_summary
from app.config import Settings
from app.graph import GraphRepository, MemoryGraphRepository
from app.models import (
    GraphEdge,
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

    findings = _detect_conflicts(repository, graph.nodes)
    reasoning_paths = _format_paths(raw_paths, graph.nodes)
    source_references = sorted({node.source_ref for node in graph.nodes if node.source_ref})
    confidence_score = _alpha(changed_node, graph.nodes, findings)
    metrics = _calculate_metrics(changed_node, graph.nodes, graph.edges, raw_paths, confidence_score)
    metrics = _adjust_risk(metrics, findings, graph.nodes)
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


def _calculate_metrics(
    changed_node: GraphNode,
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    paths: list[list[str]],
    alpha: float,
) -> MetricPayload:
    engineers = {node.owner for node in nodes if node.owner}
    teams = {node.team for node in nodes if node.team}
    depth_by_id = _depths_from_paths(changed_node.id, paths)

    gamma = 0.60
    core_hours = _bounded(changed_node.effort_hours or 16.0, 8.0, 80.0)
    downstream_hours = 0.0
    delay_by_depth: dict[int, float] = {}
    for node in nodes:
        depth = depth_by_id.get(node.id, 0 if node.id == changed_node.id else 1)
        if node.id == changed_node.id:
            effort = core_hours
        else:
            node_hours = _bounded(node.effort_hours or _default_node_hours(node), 4.0, 120.0)
            effort = node_hours * (gamma ** max(depth, 1))
            downstream_hours += effort

        allocation = 2.0
        efficiency = 0.80
        delay_by_depth[depth] = max(delay_by_depth.get(depth, 0.0), effort / (allocation * efficiency * 8.0))

    required_hours = (core_hours + downstream_hours) / max(alpha, 0.4)
    blended_rate = _blended_rate(nodes)
    risk_value = sum(_risk_value(node) for node in nodes if node.type == "RISK")
    validation_cost = 500.0 * (1.0 + math.log(1.0 / max(alpha, 0.4)))
    cost = (required_hours * blended_rate) + risk_value + validation_cost

    impacted_teams = len(teams)
    assumed_total_teams = max(impacted_teams, 4)
    beta = 0.30 * (impacted_teams / assumed_total_teams) if impacted_teams else 0.0
    delay_days = sum(delay_by_depth.values()) * (1.0 + beta)

    risk = "Low"
    if len(nodes) >= 12 or any(node.safety_critical for node in nodes) or risk_value >= 10000:
        risk = "High"
    elif len(nodes) >= 6 or risk_value > 0 or len(edges) >= 8:
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


def _alpha(changed_node: GraphNode, nodes: list[GraphNode], findings: list[ImpactFinding]) -> float:
    if not nodes:
        return 0.0
    confidence_llm = _bounded(changed_node.confidence or _confidence(nodes), 0.60, 0.98)
    anomaly_graph = 0.05
    if any(finding.severity == "critical" for finding in findings):
        anomaly_graph += 0.20
    elif any(finding.severity == "warning" for finding in findings):
        anomaly_graph += 0.10
    if len(nodes) <= 1:
        anomaly_graph += 0.05
    anomaly_graph = _bounded(anomaly_graph, 0.0, 0.40)
    return round(_bounded(confidence_llm * (1.0 - anomaly_graph), 0.40, 1.0), 2)


def _confidence(nodes: list[GraphNode]) -> float:
    return sum(node.confidence for node in nodes) / len(nodes) if nodes else 0.0


def _depths_from_paths(changed_node_id: str, paths: list[list[str]]) -> dict[str, int]:
    depths = {changed_node_id: 0}
    for path in paths:
        for depth, node_id in enumerate(path):
            if node_id not in depths or depth < depths[node_id]:
                depths[node_id] = depth
    return depths


def _default_node_hours(node: GraphNode) -> float:
    return {
        "REQUIREMENT": 16.0,
        "COMPONENT": 24.0,
        "INTERFACE": 24.0,
        "RISK": 8.0,
        "TEST": 16.0,
        "DOCUMENT": 4.0,
        "ENGINEER": 0.0,
        "TEAM": 0.0,
    }.get(node.type, 24.0)


def _blended_rate(nodes: list[GraphNode]) -> float:
    rates = [node.cost_rate for node in nodes if node.cost_rate and node.cost_rate > 0]
    if not rates:
        return 95.0
    return _bounded(sum(rates) / len(rates), 75.0, 150.0)


def _risk_value(node: GraphNode) -> float:
    value = 2500.0
    if node.safety_critical:
        value *= 1.5
    return value


def _bounded(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


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
