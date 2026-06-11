from __future__ import annotations

from neo4j import GraphDatabase

from app.config import Settings
from app.plugins.neo4jGraphPlugin.ontologyValidator import canonical_entity_type, validate_edges
from app.plugins.neo4jGraphPlugin.types import (
    ArtefactEdge,
    ArtefactNode,
    ImpactEdge,
    ImpactGraph,
    ImpactNode,
    IngestSummary,
    OntologyRule,
    OntologyValidationResult,
    ParsedArtefact,
    RequirementCandidate,
)


class Neo4jGraphRepository:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
        )
        self.database = settings.neo4j_database or None

    def close(self) -> None:
        self.driver.close()

    def health(self) -> bool:
        with self._session() as session:
            return bool(session.run("RETURN true AS ok").single()["ok"])

    def clear_graph(self) -> None:
        with self._session() as session:
            session.run("MATCH (n) WHERE n:GraphNode OR n:OntologyRule DETACH DELETE n")

    def ensure_constraints(self) -> None:
        with self._session() as session:
            session.run(
                """
                CREATE CONSTRAINT graph_node_id_unique IF NOT EXISTS
                FOR (n:GraphNode)
                REQUIRE n.id IS UNIQUE
                """
            )
            session.run(
                """
                CREATE FULLTEXT INDEX requirement_search IF NOT EXISTS
                FOR (r:Requirement)
                ON EACH [r.name, r.description]
                """
            )
            session.run("CALL db.awaitIndexes(30)")

    def ingest(self, artefact: ParsedArtefact) -> IngestSummary:
        invalid_edges = validate_edges(artefact.nodes, artefact.edges, artefact.ontology)
        self.clear_graph()
        self.ensure_constraints()
        with self._session() as session:
            for rule in artefact.ontology:
                session.run(
                    """
                    MERGE (rule:OntologyRule {
                      source_entity: $source_entity,
                      relationship: $relationship,
                      target_entity: $target_entity
                    })
                    """,
                    rule.model_dump(),
                )
            for node in artefact.nodes:
                label = canonical_entity_type(node.type)
                session.run(
                    f"""
                    MERGE (node:GraphNode:{label} {{id: $id}})
                    SET node.type = $type,
                        node.name = $name,
                        node.description = $description,
                        node.criticality = $criticality
                    """,
                    node.model_dump(),
                )
            for edge in artefact.edges:
                session.run(
                    f"""
                    MATCH (source:GraphNode {{id: $source_id}})
                    MATCH (target:GraphNode {{id: $target_id}})
                    MERGE (source)-[rel:{edge.relationship}]->(target)
                    SET rel.description = $description
                    """,
                    edge.model_dump(),
                )
        return IngestSummary(
            nodes_created=len(artefact.nodes),
            edges_created=len(artefact.edges),
            ontology_rules_created=len(artefact.ontology),
            invalid_edges=invalid_edges,
        )

    def get_ontology(self) -> list[OntologyRule]:
        with self._session() as session:
            rows = session.run(
                """
                MATCH (rule:OntologyRule)
                RETURN rule.source_entity AS source_entity,
                       rule.relationship AS relationship,
                       rule.target_entity AS target_entity
                ORDER BY source_entity, relationship, target_entity
                """
            )
            return [OntologyRule(**dict(row)) for row in rows]

    def search_requirement(self, change_text: str) -> list[RequirementCandidate]:
        with self._session() as session:
            rows = session.run(
                """
                CALL db.index.fulltext.queryNodes('requirement_search', $changeText)
                YIELD node, score
                RETURN node.id AS id,
                       node.name AS name,
                       node.description AS description,
                       node.criticality AS criticality,
                       score
                ORDER BY score DESC
                LIMIT 5
                """,
                {"changeText": change_text},
            )
            return [RequirementCandidate(**dict(row)) for row in rows]

    def get_impact_graph(self, requirement_id: str, depth: int = 2) -> ImpactGraph:
        safe_depth = max(1, min(int(depth or 2), 5))
        with self._session() as session:
            rows = session.run(
                f"""
                MATCH (rule:OntologyRule)
                WITH collect(rule.relationship) AS allowedRelationships
                MATCH path = (start:Requirement {{id: $requirementId}})-[rels*1..{safe_depth}]-(impacted:GraphNode)
                WHERE all(r IN rels WHERE type(r) IN allowedRelationships)
                RETURN path
                """,
                {"requirementId": requirement_id},
            )
            nodes: dict[str, ImpactNode] = {}
            edges: dict[str, ImpactEdge] = {}
            for row in rows:
                path = row["path"]
                for node in path.nodes:
                    node_id = node["id"]
                    nodes[node_id] = ImpactNode(
                        id=node_id,
                        label=node_id,
                        type=node.get("type", _type_from_labels(node.labels)),
                        name=node.get("name", node_id),
                        description=node.get("description", ""),
                        criticality=node.get("criticality", ""),
                        status="selected" if node_id == requirement_id else "impacted",
                    )
                for rel in path.relationships:
                    source = rel.start_node["id"]
                    target = rel.end_node["id"]
                    relationship = rel.type
                    edge_id = f"{source}->{relationship}->{target}"
                    edges[edge_id] = ImpactEdge(
                        id=edge_id,
                        source=source,
                        target=target,
                        relationship=relationship,
                        description=rel.get("description", ""),
                        status="ontology-link",
                    )
            if requirement_id not in nodes:
                node = self._get_node(requirement_id)
                if node:
                    nodes[requirement_id] = node
            return ImpactGraph(nodes=list(nodes.values()), edges=list(edges.values()))

    def validate_edges_against_ontology(self) -> OntologyValidationResult:
        with self._session() as session:
            rows = session.run(
                """
                MATCH (source:GraphNode)-[rel]->(target:GraphNode)
                WHERE NOT EXISTS {
                  MATCH (:OntologyRule {
                    source_entity: source.type,
                    relationship: type(rel),
                    target_entity: target.type
                  })
                }
                RETURN source.id AS source_id,
                       target.id AS target_id,
                       type(rel) AS relationship,
                       source.type AS source_type,
                       target.type AS target_type
                ORDER BY source_id, relationship, target_id
                """
            )
            invalid = [
                {
                    **dict(row),
                    "reason": "Relationship is not declared in the ontology sheet.",
                }
                for row in rows
            ]
            return OntologyValidationResult(valid=not invalid, invalid_edges=invalid)

    def _get_node(self, node_id: str) -> ImpactNode | None:
        with self._session() as session:
            row = session.run(
                """
                MATCH (node:GraphNode {id: $id})
                RETURN node
                LIMIT 1
                """,
                {"id": node_id},
            ).single()
            if not row:
                return None
            node = row["node"]
            return ImpactNode(
                id=node["id"],
                label=node["id"],
                type=node.get("type", _type_from_labels(node.labels)),
                name=node.get("name", node["id"]),
                description=node.get("description", ""),
                criticality=node.get("criticality", ""),
                status="selected",
            )

    def _session(self):
        return self.driver.session(database=self.database) if self.database else self.driver.session()


def _type_from_labels(labels) -> str:
    return next((label for label in labels if label not in {"GraphNode"}), "GraphNode")
