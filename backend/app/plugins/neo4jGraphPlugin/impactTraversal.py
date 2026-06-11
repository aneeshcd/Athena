from __future__ import annotations

from app.plugins.neo4jGraphPlugin.graphRepository import Neo4jGraphRepository
from app.plugins.neo4jGraphPlugin.types import ImpactGraph


def get_impact_graph(repository: Neo4jGraphRepository, requirement_id: str, depth: int = 2) -> ImpactGraph:
    return repository.get_impact_graph(requirement_id, depth)
