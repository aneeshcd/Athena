from __future__ import annotations

import re

from app.plugins.neo4jGraphPlugin.types import ArtefactEdge, ArtefactNode, InvalidEdge, OntologyRule


SAFE_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def sanitize_identifier(value: str, *, preserve_case: bool = False) -> str:
    candidate = re.sub(r"[\s-]+", "_", str(value or "").strip())
    candidate = candidate if preserve_case else candidate.upper()
    if not SAFE_IDENTIFIER.fullmatch(candidate):
        raise ValueError(f"Unsafe Neo4j label or relationship type: {value!r}")
    return candidate


def canonical_entity_type(value: str) -> str:
    candidate = sanitize_identifier(value, preserve_case=True)
    return "".join(part[:1].upper() + part[1:].lower() for part in candidate.split("_") if part)


def canonical_relationship(value: str) -> str:
    return sanitize_identifier(value)


def validate_edges(
    nodes: list[ArtefactNode],
    edges: list[ArtefactEdge],
    ontology: list[OntologyRule],
) -> list[InvalidEdge]:
    node_types = {node.id: node.type for node in nodes}
    allowed = {
        (rule.source_entity, rule.relationship, rule.target_entity)
        for rule in ontology
    }
    invalid: list[InvalidEdge] = []
    for edge in edges:
        source_type = node_types.get(edge.source_id)
        target_type = node_types.get(edge.target_id)
        if not source_type or not target_type:
            invalid.append(
                InvalidEdge(
                    source_id=edge.source_id,
                    target_id=edge.target_id,
                    relationship=edge.relationship,
                    source_type=source_type,
                    target_type=target_type,
                    reason="Source or target node is missing from the nodes sheet.",
                )
            )
            continue
        if (source_type, edge.relationship, target_type) not in allowed:
            invalid.append(
                InvalidEdge(
                    source_id=edge.source_id,
                    target_id=edge.target_id,
                    relationship=edge.relationship,
                    source_type=source_type,
                    target_type=target_type,
                    reason="Relationship is not declared in the ontology sheet.",
                )
            )
    return invalid
