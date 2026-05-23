from __future__ import annotations

import json
import re
from collections import deque
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
        with self.driver.session() as session:
            if exact_id:
                row = session.run(
                    "MATCH (n:EngineeringEntity {id: $id}) RETURN n LIMIT 1",
                    {"id": exact_id},
                ).single()
                if row:
                    return self._node_from_record(row["n"])

            terms = [term.lower() for term in re.findall(r"[a-zA-Z0-9_-]{3,}", change_request)[:12]]
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
                     ) AS score
                WHERE score > 0
                RETURN n, score
                ORDER BY score DESC, n.confidence DESC
                LIMIT 1
                """,
                {"terms": terms},
            ).single()
            return self._node_from_record(row["n"]) if row else None

    def downstream_impact(self, node_id: str, max_depth: int = 4) -> tuple[GraphPayload, list[list[str]]]:
        with self.driver.session() as session:
            rows = session.run(
                """
                MATCH path = (start:EngineeringEntity {id: $node_id})-[*1..4]-(end:EngineeringEntity)
                WHERE all(rel IN relationships(path) WHERE type(rel) IN [
                  'DEPENDS_ON', 'VALIDATES', 'CONFLICTS_WITH', 'DERIVED_FROM',
                  'IMPLEMENTS', 'OWNED_BY', 'BELONGS_TO', 'MITIGATES', 'AFFECTS',
                  'SEMANTICALLY_SIMILAR'
                ])
                RETURN path
                LIMIT 250
                """,
                {"node_id": node_id, "max_depth": max_depth},
            )
            node_by_id: dict[str, GraphNode] = {}
            edge_by_id: dict[str, GraphEdge] = {}
            paths: list[list[str]] = []

            for row in rows:
                path = row["path"]
                node_ids: list[str] = []
                for node in path.nodes:
                    graph_node = self._node_from_record(node)
                    node_by_id[graph_node.id] = graph_node
                    node_ids.append(graph_node.id)
                paths.append(node_ids)
                for relationship in path.relationships:
                    source = relationship.start_node["id"]
                    target = relationship.end_node["id"]
                    edge = GraphEdge(
                        id=f"{source}->{relationship.type}->{target}",
                        source=source,
                        target=target,
                        type=relationship.type,
                        rationale=relationship.get("rationale", ""),
                        confidence=relationship.get("confidence", 0.75),
                    )
                    edge_by_id[edge.id] = edge

        if node_id not in node_by_id:
            matched = self.match_change_node(node_id)
            if matched:
                node_by_id[matched.id] = matched

        return GraphPayload(nodes=list(node_by_id.values()), edges=list(edge_by_id.values())), paths

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
            safety_critical=entity.safety_critical,
            confidence=entity.confidence,
        )

    def downstream_impact(self, node_id: str, max_depth: int = 4) -> tuple[GraphPayload, list[list[str]]]:
        adjacency: dict[str, list[str]] = {}
        for rel in self.relationships:
            adjacency.setdefault(rel.source_id, []).append(rel.target_id)
            adjacency.setdefault(rel.target_id, []).append(rel.source_id)

        visited = {node_id}
        queue = deque([(node_id, [node_id], 0)])
        paths: list[list[str]] = []
        while queue:
            current, path, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for target in adjacency.get(current, []):
                if target in visited:
                    continue
                visited.add(target)
                next_path = path + [target]
                paths.append(next_path)
                queue.append((target, next_path, depth + 1))

        graph = self.get_graph()
        nodes = [node for node in graph.nodes if node.id in visited]
        edges = [edge for edge in graph.edges if edge.source in visited and edge.target in visited]
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
