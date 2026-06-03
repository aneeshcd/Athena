from __future__ import annotations

import json
import math
import re
from collections import deque
from typing import Iterable
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError

from app.ai import embed_texts
from app.config import Settings
from app.models import (
    EngineeringEntity,
    EngineeringRelationship,
    GraphEdge,
    GraphNode,
    GraphPayload,
    NormalizedArtifact,
)


class GraphRepository:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )

    def close(self) -> None:
        self.driver.close()

    def health(self) -> bool:
        with self.driver.session() as session:
            return bool(session.run("RETURN true AS ok").single()["ok"])

    def ensure_constraints(self) -> None:
        statements = [
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:EngineeringEntity) REQUIRE e.id IS UNIQUE",
            "CREATE INDEX entity_type IF NOT EXISTS FOR (e:EngineeringEntity) ON (e.type)",
            "CREATE INDEX entity_title IF NOT EXISTS FOR (e:EngineeringEntity) ON (e.title)",
        ]
        with self.driver.session() as session:
            for statement in statements:
                session.run(statement)

    def clear_graph(self) -> None:
        with self.driver.session() as session:
            session.run("MATCH (n:EngineeringEntity) DETACH DELETE n")

    def ingest_artifact(self, artifact: NormalizedArtifact) -> list[str]:
        self.ensure_constraints()
        descriptions = [f"{entity.title}\n{entity.description}" for entity in artifact.entities]
        embeddings = embed_texts(descriptions, self.settings)
        cypher_statements: list[str] = []

        with self.driver.session() as session:
            for entity, embedding in zip(artifact.entities, embeddings, strict=False):
                statement, parameters = self._merge_entity_statement(entity, embedding)
                cypher_statements.append(statement)
                session.run(statement, parameters)

            for relationship in artifact.relationships:
                statement, parameters = self._merge_relationship_statement(relationship)
                cypher_statements.append(statement)
                session.run(statement, parameters)

        return cypher_statements

    def get_graph(self, limit: int = 250) -> GraphPayload:
        with self.driver.session() as session:
            node_rows = session.run(
                """
                MATCH (n:EngineeringEntity)
                RETURN n
                ORDER BY n.type, n.id
                LIMIT $limit
                """,
                {"limit": limit},
            )
            nodes = [self._node_from_record(row["n"]) for row in node_rows]

            edge_rows = session.run(
                """
                MATCH (a:EngineeringEntity)-[r]->(b:EngineeringEntity)
                RETURN a.id AS source, b.id AS target, type(r) AS type,
                       coalesce(r.rationale, '') AS rationale,
                       coalesce(r.confidence, 0.75) AS confidence
                LIMIT $limit
                """,
                {"limit": limit * 2},
            )
            edges = [
                GraphEdge(
                    id=f"{row['source']}->{row['type']}->{row['target']}",
                    source=row["source"],
                    target=row["target"],
                    type=row["type"],
                    rationale=row["rationale"],
                    confidence=row["confidence"],
                )
                for row in edge_rows
            ]

        return GraphPayload(nodes=nodes, edges=edges)

    def match_change_node(self, change_request: str) -> GraphNode | None:
        exact_id = self._extract_id(change_request)
        query_embedding = embed_texts([change_request], self.settings)[0]
        with self.driver.session() as session:
            if exact_id:
                row = session.run(
                    "MATCH (n:EngineeringEntity {id: $id}) RETURN n LIMIT 1",
                    {"id": exact_id},
                ).single()
                if row:
                    return self._node_from_record(row["n"])

            terms = [term.lower() for term in re.findall(r"[a-zA-Z0-9_-]{3,}", change_request)[:18]]
            row = session.run(
                """
                MATCH (n:EngineeringEntity)
                WITH n,
                     reduce(score = 0, term IN $terms |
                       score + CASE
                         WHEN toLower(n.id) CONTAINS term THEN 5
                         WHEN toLower(n.title) CONTAINS term THEN 3
                         WHEN toLower(n.description) CONTAINS term THEN 1
                         ELSE 0
                       END
                     ) AS lexical_score
                RETURN n, lexical_score, coalesce(n.embedding, []) AS embedding
                ORDER BY lexical_score DESC, n.confidence DESC
                LIMIT 80
                """,
                {"terms": terms},
            )
            best_node = None
            best_score = -1.0
            for candidate in row:
                lexical_score = float(candidate["lexical_score"] or 0)
                semantic_score = _cosine_similarity(query_embedding, candidate["embedding"])
                confidence = float(candidate["n"].get("confidence", 0.75))
                score = lexical_score + (semantic_score * 4.0) + confidence
                if score > best_score:
                    best_score = score
                    best_node = candidate["n"]
            return self._node_from_record(best_node) if best_node is not None else None

    def downstream_impact(self, node_id: str, max_depth: int = 4) -> tuple[GraphPayload, list[list[str]]]:
        max_depth = max(1, min(max_depth, 5))
        with self.driver.session() as session:
            node_rows = session.run("MATCH (n:EngineeringEntity) RETURN n")
            node_by_id = {
                graph_node.id: graph_node
                for graph_node in (self._node_from_record(row["n"]) for row in node_rows)
            }

            edge_rows = session.run(
                """
                MATCH (a:EngineeringEntity)-[r]->(b:EngineeringEntity)
                WHERE type(r) IN [
                  'DEPENDS_ON', 'VALIDATES', 'CONFLICTS_WITH',
                  'IMPLEMENTS', 'MITIGATES', 'AFFECTS', 'SEMANTICALLY_SIMILAR'
                ]
                RETURN a.id AS source, b.id AS target, type(r) AS type,
                       coalesce(r.rationale, '') AS rationale,
                       coalesce(r.confidence, 0.75) AS confidence
                """
            )
            edge_by_id = {
                f"{row['source']}->{row['type']}->{row['target']}": GraphEdge(
                    id=f"{row['source']}->{row['type']}->{row['target']}",
                    source=row["source"],
                    target=row["target"],
                    type=row["type"],
                    rationale=row["rationale"],
                    confidence=row["confidence"],
                )
                for row in edge_rows
            }

        if node_id not in node_by_id:
            matched = self.match_change_node(node_id)
            if matched:
                node_by_id[matched.id] = matched

        impacted_ids, impacted_edges, paths = _traverse_impacts(node_id, edge_by_id.values(), max_depth)
        impacted_ids.add(node_id)
        return (
            GraphPayload(
                nodes=[node for node_id, node in node_by_id.items() if node_id in impacted_ids],
                edges=impacted_edges,
            ),
            paths,
        )

    def orphan_requirements(self, node_ids: list[str]) -> list[str]:
        if not node_ids:
            return []
        with self.driver.session() as session:
            rows = session.run(
                """
                MATCH (r:EngineeringEntity)
                WHERE r.id IN $node_ids AND r.type = 'REQUIREMENT'
                OPTIONAL MATCH (:EngineeringEntity)-[v:VALIDATES]->(r)
                WITH r, count(v) AS validations
                WHERE validations = 0
                RETURN r.id AS id
                """,
                {"node_ids": node_ids},
            )
            return [row["id"] for row in rows]

    def _merge_entity_statement(self, entity: EngineeringEntity, embedding: list[float]) -> tuple[str, dict]:
        statement = """
        MERGE (e:EngineeringEntity {id: $id})
        SET e.type = $type,
            e.title = $title,
            e.description = $description,
            e.source_ref = $source_ref,
            e.author = $author,
            e.owner = $owner,
            e.team = $team,
            e.artifact_id = $artifact_id,
            e.timestamp = $timestamp,
            e.effort_hours = $effort_hours,
            e.cost_rate = $cost_rate,
            e.delay_days = $delay_days,
            e.safety_critical = $safety_critical,
            e.confidence = $confidence,
            e.metadata_json = $metadata_json,
            e.embedding = $embedding
        """
        return statement, {
            "id": entity.id,
            "type": entity.type,
            "title": entity.title,
            "description": entity.description,
            "source_ref": entity.source_ref,
            "author": entity.author,
            "owner": entity.owner,
            "team": entity.team,
            "artifact_id": entity.artifact_id,
            "timestamp": entity.timestamp.isoformat(),
            "effort_hours": entity.effort_hours,
            "cost_rate": entity.cost_rate,
            "delay_days": entity.delay_days,
            "safety_critical": entity.safety_critical,
            "confidence": entity.confidence,
            "metadata_json": json.dumps(entity.metadata, default=str),
            "embedding": embedding,
        }

    def _merge_relationship_statement(self, relationship: EngineeringRelationship) -> tuple[str, dict]:
        statement = f"""
        MATCH (source:EngineeringEntity {{id: $source_id}})
        MATCH (target:EngineeringEntity {{id: $target_id}})
        MERGE (source)-[r:{relationship.type}]->(target)
        SET r.rationale = $rationale,
            r.confidence = $confidence
        """
        return statement, relationship.model_dump()

    def _node_from_record(self, record) -> GraphNode:
        return GraphNode(
            id=record["id"],
            label=record.get("title", record["id"]),
            type=record.get("type", "REQUIREMENT"),
            source_ref=record.get("source_ref"),
            owner=record.get("owner"),
            team=record.get("team"),
            effort_hours=record.get("effort_hours", 8.0),
            cost_rate=record.get("cost_rate", 95.0),
            delay_days=record.get("delay_days", 0.5),
            safety_critical=record.get("safety_critical", False),
            confidence=record.get("confidence", 0.75),
        )

    def _extract_id(self, text: str) -> str | None:
        dotted = re.search(r"\bG\.D\d+(?:\.\d+)+\b", text, re.IGNORECASE)
        if dotted:
            return dotted.group(0).upper()
        match = re.search(r"\b(REQ|TEST|COMP|IF|RISK|SYS|SW|HW|DOC)[-_ ][A-Z0-9]{2,12}\b", text, re.IGNORECASE)
        return match.group(0).replace(" ", "-").upper() if match else None


class MemoryGraphRepository:
    """Small development fallback used only when Neo4j is unavailable."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.nodes: dict[str, EngineeringEntity] = {}
        self.relationships: list[EngineeringRelationship] = []

    def close(self) -> None:
        return None

    def health(self) -> bool:
        return True

    def ensure_constraints(self) -> None:
        return None

    def clear_graph(self) -> None:
        self.nodes.clear()
        self.relationships.clear()

    def ingest_artifact(self, artifact: NormalizedArtifact) -> list[str]:
        for entity in artifact.entities:
            self.nodes[entity.id] = entity
        self.relationships.extend(artifact.relationships)
        return ["// Neo4j unavailable; stored artifact in memory for this backend process."]

    def get_graph(self, limit: int = 250) -> GraphPayload:
        nodes = [
            GraphNode(
                id=entity.id,
                label=entity.title,
                type=entity.type,
                source_ref=entity.source_ref,
                owner=entity.owner,
                team=entity.team,
                effort_hours=entity.effort_hours,
                cost_rate=entity.cost_rate,
                delay_days=entity.delay_days,
                safety_critical=entity.safety_critical,
                confidence=entity.confidence,
            )
            for entity in list(self.nodes.values())[:limit]
        ]
        edges = [
            GraphEdge(
                id=f"{rel.source_id}->{rel.type}->{rel.target_id}",
                source=rel.source_id,
                target=rel.target_id,
                type=rel.type,
                rationale=rel.rationale,
                confidence=rel.confidence,
            )
            for rel in self.relationships[: limit * 2]
        ]
        return GraphPayload(nodes=nodes, edges=edges)

    def match_change_node(self, change_request: str) -> GraphNode | None:
        exact_id = self._extract_id(change_request)
        if exact_id and exact_id in self.nodes:
            entity = self.nodes[exact_id]
            return GraphNode(
                id=entity.id,
                label=entity.title,
                type=entity.type,
                source_ref=entity.source_ref,
                owner=entity.owner,
                team=entity.team,
                effort_hours=entity.effort_hours,
                cost_rate=entity.cost_rate,
                delay_days=entity.delay_days,
                safety_critical=entity.safety_critical,
                confidence=entity.confidence,
            )
        terms = set(re.findall(r"[a-zA-Z0-9_-]{3,}", change_request.lower()))
        best: tuple[int, EngineeringEntity] | None = None
        for entity in self.nodes.values():
            haystack = f"{entity.id} {entity.title} {entity.description}".lower()
            score = sum(1 for term in terms if term in haystack)
            if score and (best is None or score > best[0]):
                best = (score, entity)
        if not best and self.nodes:
            best = (0, next(iter(self.nodes.values())))
        if not best:
            return None
        entity = best[1]
        return GraphNode(
            id=entity.id,
            label=entity.title,
            type=entity.type,
            source_ref=entity.source_ref,
            owner=entity.owner,
            team=entity.team,
            effort_hours=entity.effort_hours,
            cost_rate=entity.cost_rate,
            delay_days=entity.delay_days,
            safety_critical=entity.safety_critical,
            confidence=entity.confidence,
        )

    def downstream_impact(self, node_id: str, max_depth: int = 4) -> tuple[GraphPayload, list[list[str]]]:
        graph = self.get_graph()
        visited, edges, paths = _traverse_impacts(node_id, graph.edges, max_depth)
        visited.add(node_id)
        nodes = [node for node in graph.nodes if node.id in visited]
        return GraphPayload(nodes=nodes, edges=edges), paths

    def orphan_requirements(self, node_ids: list[str]) -> list[str]:
        validated = {rel.target_id for rel in self.relationships if rel.type == "VALIDATES"}
        return [
            entity.id
            for entity in self.nodes.values()
            if entity.id in node_ids and entity.type == "REQUIREMENT" and entity.id not in validated
        ]

    def _extract_id(self, text: str) -> str | None:
        dotted = re.search(r"\bG\.D\d+(?:\.\d+)+\b", text, re.IGNORECASE)
        if dotted:
            return dotted.group(0).upper()
        match = re.search(r"\b(REQ|TEST|COMP|IF|RISK|SYS|SW|HW|DOC)[-_ ][A-Z0-9]{2,12}\b", text, re.IGNORECASE)
        return match.group(0).replace(" ", "-").upper() if match else None


def create_repository(settings: Settings) -> GraphRepository | MemoryGraphRepository:
    try:
        repository = GraphRepository(settings)
        repository.health()
        return repository
    except Neo4jError:
        return MemoryGraphRepository(settings)
    except Exception:
        return MemoryGraphRepository(settings)


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _traverse_impacts(
    start_id: str,
    edges: Iterable[GraphEdge],
    max_depth: int,
) -> tuple[set[str], list[GraphEdge], list[list[str]]]:
    adjacency: dict[str, list[tuple[str, GraphEdge]]] = {}
    edge_by_step: dict[str, GraphEdge] = {}
    for edge in edges:
        for source, target in _impact_steps(edge):
            adjacency.setdefault(source, []).append((target, edge))
            edge_by_step[f"{source}->{target}"] = edge

    visited = {start_id}
    queue = deque([(start_id, [start_id], 0)])
    paths: list[list[str]] = []
    while queue:
        current, path, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for target, _edge in adjacency.get(current, []):
            if target in visited:
                continue
            visited.add(target)
            next_path = path + [target]
            paths.append(next_path)
            queue.append((target, next_path, depth + 1))

    impacted_edge_by_id: dict[str, GraphEdge] = {}
    for path in paths:
        for source, target in zip(path, path[1:], strict=False):
            edge = edge_by_step.get(f"{source}->{target}")
            if edge:
                impacted_edge_by_id[edge.id] = edge
    return visited, list(impacted_edge_by_id.values()), paths


def _impact_steps(edge: GraphEdge) -> list[tuple[str, str]]:
    if edge.type == "VALIDATES":
        return [(edge.target, edge.source)]
    if edge.type in {"CONFLICTS_WITH", "SEMANTICALLY_SIMILAR"}:
        return [(edge.source, edge.target), (edge.target, edge.source)]
    if edge.type in {"DEPENDS_ON", "IMPLEMENTS", "MITIGATES", "AFFECTS"}:
        return [(edge.source, edge.target)]
    return []
