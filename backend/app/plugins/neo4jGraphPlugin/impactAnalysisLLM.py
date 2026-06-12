from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.request
from typing import Any

from openai import OpenAI
from pydantic import ValidationError

from app.config import Settings, get_settings
from app.plugins.neo4jGraphPlugin.types import (
    ImpactGraph,
    LLMImpactAnalysisInput,
    LLMImpactAnalysisResult,
    RippleEffect,
    SelectedRequirement,
)


HITL_NOTICE = (
    "This AI analysis is advisory only. The responsible engineer must review and approve any "
    "requirement, design, test, or certification action."
)
logger = logging.getLogger(__name__)
FAST_MODEL_CANDIDATES = ("qwen2.5:3b", "llama3.2:1b")
GRAPH_SPECIFIC_TYPES = {"Subsystem", "SoftwareModule", "TestCase", "Test", "Risk", "Issue", "Document", "Requirement"}
TYPE_PRIORITY = {
    "Requirement": 0,
    "Subsystem": 1,
    "SoftwareModule": 2,
    "TestCase": 3,
    "Test": 3,
    "Risk": 4,
    "Issue": 5,
    "Document": 6,
    "Team": 7,
    "Person": 8,
}
CRITICALITY_PRIORITY = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
MAX_RIPPLE_EFFECTS = 3
FACT_CATEGORIES = {
    "Requirement": "connectedRequirements",
    "Subsystem": "connectedSubsystems",
    "SoftwareModule": "connectedSoftwareModules",
    "TestCase": "connectedTestCases",
    "Test": "connectedTestCases",
    "Risk": "connectedRisks",
    "Issue": "connectedIssues",
    "Document": "connectedDocuments",
    "Team": "connectedTeams",
}
PROVIDER_WORD_PATTERN = re.compile(r"\b(?:Ollama|OpenAI|provider|model|fallback|llama[-_.:\w]*|gpt[-_.:\w]*)\b", re.IGNORECASE)


class ImpactAnalysisUnavailable(RuntimeError):
    pass


class ImpactAnalysisFailed(RuntimeError):
    pass


def generateImpactAnalysis(
    input: LLMImpactAnalysisInput,
    settings: Settings | None = None,
    client: Any | None = None,
    ollama_client: Any | None = None,
) -> LLMImpactAnalysisResult:
    settings = settings or get_settings()
    provider = (settings.llm_provider or "ollama").lower()
    node_count = len(input.impactGraph.nodes)
    edge_count = len(input.impactGraph.edges)
    logger.info(
        "[LLM] Generating impact analysis using provider=%s openaiConfigured=%s openaiModel=%s ollamaBaseUrl=%s ollamaModel=%s selectedRequirementPresent=%s nodeCount=%s edgeCount=%s",
        provider,
        bool(settings.openai_api_key),
        settings.openai_model or "gpt-4.1-mini",
        settings.ollama_base_url,
        settings.ollama_model,
        input.selectedRequirement is not None,
        node_count,
        edge_count,
    )
    if not input.impactGraph.nodes:
        logger.warning("AI impact analysis rejected: empty impact graph.")
        raise ValueError("Impact graph is empty; AI impact analysis requires computed graph data.")
    if provider not in {"auto", "openai", "ollama", "fallback"}:
        logger.warning("[LLM] Unknown LLM_PROVIDER=%s; defaulting to Ollama.", provider)
        provider = "ollama"

    if provider in {"auto", "openai"}:
        if settings.openai_api_key or client is not None:
            try:
                return _generate_with_openai(input, settings, client)
            except Exception as exc:
                if provider == "openai":
                    reason = _safe_error_message(exc)
                    logger.warning("[LLM] Falling back to rule-based provider because: %s", reason)
                    return _generate_fallback(input, reason)
                logger.warning("[LLM] OpenAI provider failed; falling back to Ollama. reason=%s", _safe_error_message(exc))
        elif provider == "openai":
            logger.warning("[LLM] Falling back to rule-based provider because: OPENAI_API_KEY is not configured.")
            return _generate_fallback(input, "OPENAI_API_KEY is not configured.")
        else:
            logger.info("[LLM] OpenAI provider skipped: OPENAI_API_KEY is not configured.")

    if provider in {"auto", "ollama"}:
        try:
            return _generate_with_ollama(input, settings, ollama_client)
        except Exception as exc:
            logger.warning("[LLM] Falling back to rule-based provider because: %s", _safe_error_message(exc))
            return _generate_fallback(input, _safe_error_message(exc))

    return _generate_fallback(input, "No LLM provider returned a valid analysis.")


def _generate_with_openai(
    input: LLMImpactAnalysisInput,
    settings: Settings,
    client: Any | None,
) -> LLMImpactAnalysisResult:
    model = settings.openai_model or "gpt-4.1-mini"
    openai_client = client or OpenAI(api_key=settings.openai_api_key, timeout=30)
    try:
        logger.info("[LLM] Calling OpenAI model=%s", model)
        response = openai_client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=_messages(_compact_context(input.changeText, input.selectedRequirement, input.impactGraph)),
        )
        logger.info("[LLM] OpenAI response received")
        content = response.choices[0].message.content
    except Exception as exc:
        logger.exception("[LLM] OpenAI request failed: %s", _safe_error_message(exc))
        raise ImpactAnalysisFailed(f"OpenAI request failed: {_safe_error_message(exc)}") from exc
    analysis = _parse_analysis(content, "openai")
    logger.info("[LLM] OpenAI JSON parsed successfully")
    return _finalize_analysis(analysis, input.impactGraph, "openai", input.selectedRequirement)


def _generate_with_ollama(
    input: LLMImpactAnalysisInput,
    settings: Settings,
    ollama_client: Any | None,
) -> LLMImpactAnalysisResult:
    health = _ollama_tags_health(settings)
    logger.info("[LLM][Ollama] health check started")
    logger.info(
        "[LLM][Ollama] health check ok=%s in %s ms",
        health["reachable"],
        health["tagsLatencyMs"],
    )
    logger.info("[LLM][Ollama] model found: %s", health["modelAvailable"])
    context = _compact_context(
        input.changeText,
        input.selectedRequirement,
        input.impactGraph,
        settings.ollama_max_context_nodes,
        settings.ollama_max_context_edges,
        settings.ollama_max_description_chars,
    )
    _log_prompt_diagnostics(context)
    logger.info(
        "[LLM] Calling Ollama at %s/api/chat model=%s",
        settings.ollama_base_url,
        settings.ollama_model,
    )
    try:
        first_content = _call_ollama(settings, context, strict=False, ollama_client=ollama_client)
        analysis = _parse_analysis(first_content, "ollama", input)
        logger.info("[LLM] Ollama JSON parsed successfully")
        if _all_ripple_effects_generic(analysis, input.impactGraph):
            logger.warning("[LLM][Ollama] ripple effects were generic; retrying once with stricter graph-specific prompt.")
            raise ImpactAnalysisFailed("Generic ripple effects returned by LLM.")
        return _finalize_ollama_analysis(analysis, input, context)
    except ImpactAnalysisFailed as exc:
        logger.warning("[LLM][Ollama] first attempt failed; retrying once with shorter prompt. reason=%s", _safe_error_message(exc))
    retry_context = _compact_context(
        input.changeText,
        input.selectedRequirement,
        input.impactGraph,
        max(5, min(settings.ollama_max_context_nodes, 10)),
        max(8, min(settings.ollama_max_context_edges, 15)),
        0,
    )
    _log_prompt_diagnostics(retry_context)
    retry_content = _call_ollama(settings, retry_context, strict=True, ollama_client=ollama_client)
    analysis = _parse_analysis(retry_content, "ollama", input)
    logger.info("[LLM] Ollama JSON parsed successfully")
    return _finalize_ollama_analysis(analysis, input, retry_context)


def _call_ollama(
    settings: Settings,
    context: dict[str, Any],
    strict: bool,
    ollama_client: Any | None,
) -> str:
    payload = {
        "model": settings.ollama_model,
        "messages": _messages(context, strict=strict),
        "stream": False,
        "format": "json",
        "keep_alive": "5m",
        "options": {"temperature": 0, "num_predict": 900, "num_ctx": 2048},
    }
    timeout_seconds = max(1, int(settings.ollama_timeout_ms / 1000))
    started = time.perf_counter()
    prompt_content = payload["messages"][1]["content"]
    logger.info("[LLM][Ollama] generation started")
    logger.info("[LLM][Ollama] prompt chars: %s", len(prompt_content))
    logger.info("[LLM][Ollama] approximate tokens: %s", max(1, len(prompt_content) // 4))
    if ollama_client is not None:
        try:
            response = ollama_client(payload)
        except Exception as exc:
            duration_ms = _elapsed_ms(started)
            logger.exception("[LLM][Ollama] generation failed in %s ms: %s", duration_ms, _safe_error_message(exc))
            raise ImpactAnalysisFailed(f"Ollama request failed: {_safe_error_message(exc)}") from exc
    else:
        url = settings.ollama_base_url.rstrip("/") + "/api/chat"
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as handle:
                response = json.loads(handle.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            duration_ms = _elapsed_ms(started)
            logger.exception("[LLM][Ollama] generation failed in %s ms: %s", duration_ms, _safe_error_message(exc))
            logger.exception("[LLM] Ollama request failed: %s", _safe_error_message(exc))
            raise ImpactAnalysisFailed(f"Ollama request failed: {_safe_error_message(exc)}") from exc
    duration_ms = _elapsed_ms(started)
    logger.info("[LLM][Ollama] generation completed in %s ms", duration_ms)
    logger.info("[LLM] Ollama response received")
    content = response.get("message", {}).get("content") if isinstance(response, dict) else None
    if not content:
        raise ImpactAnalysisFailed("Ollama returned an empty impact analysis response.")
    return content


def get_llm_health(settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    provider = (settings.llm_provider or "ollama").lower()
    if provider not in {"auto", "openai", "ollama", "fallback"}:
        provider = "ollama"
    ollama = _ollama_tags_health(settings)
    return {
        "provider": provider,
        "ollama": ollama,
        "openai": {"configured": bool(settings.openai_api_key)},
        "config": _ollama_config(settings),
    }


def test_ollama_generation(prompt: str, settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    started = time.perf_counter()
    try:
        content = _call_ollama_prompt(settings, prompt)
        duration_ms = _elapsed_ms(started)
        parsed = None
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = None
        return {
            "provider": "ollama",
            "success": True,
            "durationMs": duration_ms,
            "rawResponsePreview": content[:500],
            "parsed": parsed,
        }
    except Exception as exc:
        return {
            "provider": "ollama",
            "success": False,
            "durationMs": _elapsed_ms(started),
            "error": _safe_error_message(exc),
        }


def _ollama_tags_health(settings: Settings) -> dict[str, Any]:
    started = time.perf_counter()
    logger.info("[LLM][Ollama] health check started")
    ollama = {
        "baseUrl": settings.ollama_base_url,
        "model": settings.ollama_model,
        "reachable": False,
        "modelAvailable": False,
        "tagsLatencyMs": None,
    }
    try:
        url = settings.ollama_base_url.rstrip("/") + "/api/tags"
        request = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(request, timeout=10) as handle:
            payload = json.loads(handle.read().decode("utf-8"))
        model_names = {
            model.get("name")
            for model in payload.get("models", [])
            if isinstance(model, dict)
        }
        ollama["reachable"] = True
        ollama["modelAvailable"] = settings.ollama_model in model_names
        ollama["availableModels"] = sorted(model_names)
        ollama["tagsLatencyMs"] = _elapsed_ms(started)
        logger.info(
            "[LLM][Ollama] health check ok in %s ms",
            ollama["tagsLatencyMs"],
        )
        logger.info(
            "[LLM][Ollama] model found: %s baseUrl=%s model=%s",
            ollama["modelAvailable"],
            settings.ollama_base_url,
            settings.ollama_model,
        )
    except Exception as exc:
        ollama["tagsLatencyMs"] = _elapsed_ms(started)
        logger.warning("[LLM][Ollama] health check failed in %s ms: %s", ollama["tagsLatencyMs"], _safe_error_message(exc))
    return ollama


def _call_ollama_prompt(settings: Settings, prompt: str) -> str:
    payload = {
        "model": settings.ollama_model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": "json",
        "keep_alive": "5m",
        "options": {"temperature": 0, "num_predict": 900, "num_ctx": 1024},
    }
    timeout_seconds = max(1, int(settings.ollama_timeout_ms / 1000))
    url = settings.ollama_base_url.rstrip("/") + "/api/chat"
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout_seconds) as handle:
        response = json.loads(handle.read().decode("utf-8"))
    content = response.get("message", {}).get("content") if isinstance(response, dict) else None
    if not content:
        raise ImpactAnalysisFailed("Ollama returned an empty generation test response.")
    return content


def _ollama_config(settings: Settings) -> dict[str, Any]:
    return {
        "timeoutMs": settings.ollama_timeout_ms,
        "maxContextNodes": settings.ollama_max_context_nodes,
        "maxContextEdges": settings.ollama_max_context_edges,
        "maxDescriptionChars": settings.ollama_max_description_chars,
        "warmupOnStart": settings.ollama_warmup_on_start,
    }


def _generate_fallback(input: LLMImpactAnalysisInput, reason: str = "LLM provider unavailable.") -> LLMImpactAnalysisResult:
    logger.info("[LLM] Falling back to rule-based provider because: %s", reason)
    impacted_nodes = [node for node in input.impactGraph.nodes if node.status != "selected"]
    direct_nodes = [node for node in impacted_nodes if node.hop in {None, 1}]
    affected_ids = [node.id for node in impacted_nodes]
    affected_types = sorted({node.type for node in impacted_nodes if node.type})
    type_phrase = ", ".join(affected_types) if affected_types else "connected graph elements"
    analysis = LLMImpactAnalysisResult(
        provider="fallback",
        summary=(
            f"{_summary_prefix(input.selectedRequirement, input.impactGraph)} "
            f"The graph contains {len(impacted_nodes)} connected element(s), including {type_phrase}."
        ),
        rippleEffects=_graph_derived_ripple_effects(input.impactGraph),
        suggestedNextSteps=_graph_derived_next_steps(input.selectedRequirement, input.impactGraph),
        engineeringReviewChecklist=[
            "Confirm the selected requirement is correct.",
            "Confirm the impacted subsystem and software modules are relevant.",
            "Review linked test cases.",
            "Review linked risks and issues.",
            "Record the engineer's final decision.",
        ],
        assumptionsAndLimitations=[
            "This fallback analysis is rule-based and does not use an LLM.",
            f"Fallback reason: {reason}",
            "It is based only on the provided graph output.",
        ],
        humanInTheLoopNotice=HITL_NOTICE,
    )
    return _finalize_analysis(analysis, input.impactGraph, "fallback", input.selectedRequirement)


def _messages(context: dict[str, Any], strict: bool = False) -> list[dict[str, str]]:
    system_content = (
        "You are an engineering impact-analysis assistant. Use only the provided selected requirement, "
        "impacted nodes, and relationships. Generate requirement-specific ripple effects from real graph areas. "
        "Do not invent nodes, systems, tests, risks, issues, documents, software modules, subsystems, or relationships. "
        "Return only valid JSON. No markdown. No comments. No text outside JSON. "
        "Do not mention Ollama, OpenAI, provider, model, or fallback."
    )
    if strict:
        system_content += (
            " Return a complete compact JSON object only. The previous response was too generic. "
            "Use real node names from the graph in the summary and ripple effect areas."
        )
    return [
        {"role": "system", "content": system_content},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": "Generate concise JSON using exactly the requested schema.",
                    "rules": [
                        "Use only the provided selectedRequirement, impactedNodes, impactedRelationships, impactFacts, and coverageFlags.",
                        "Do not decide what is impacted; Neo4j impactFacts are the source of truth.",
                        "Do not invent node IDs, names, or relationships.",
                        "Generate 2 to 3 rippleEffects when the graph has enough connected nodes.",
                        "Generate 1 rippleEffect only when the graph is very small.",
                        "Each ripple effect must reference a real area from the provided graph, such as a subsystem, software module, test case, risk, issue, document, or related requirement.",
                        "Do not use generic area names like Knowledge graph impact unless there is no better graph-specific area.",
                        "Use provided node IDs and node names where helpful.",
                        "Use different ripple effect areas when possible.",
                        "Do not repeat the same point.",
                        "Minimum 1 rippleEffect when graph contains impacted nodes.",
                        "Do not make final decisions.",
                        "Return valid JSON only.",
                        "Do not include markdown.",
                        "Do not include trailing commas.",
                        "Return a complete JSON object.",
                        f"Maximum {MAX_RIPPLE_EFFECTS} rippleEffects.",
                        "Maximum 5 suggestedNextSteps.",
                        "Keep each explanation under 35 words.",
                        "Do not mention Ollama, OpenAI, provider, model, or fallback.",
                    ],
                    **context,
                    "requiredOutputSchema": {
                        "summary": "string",
                        "rippleEffects": [
                            {"area": "string", "explanation": "string"}
                        ],
                        "suggestedNextSteps": ["string"],
                    },
                },
                ensure_ascii=True,
            ),
        },
    ]


def _compact_context(
    change_text: str,
    selected_requirement: SelectedRequirement,
    impact_graph: ImpactGraph,
    max_nodes: int | None = None,
    max_edges: int | None = None,
    max_description_chars: int = 180,
) -> dict[str, Any]:
    nodes = _sorted_nodes(impact_graph.nodes, selected_requirement.id)
    edges = _sorted_edges(impact_graph.edges)
    total_nodes = len(nodes)
    total_edges = len(edges)
    limited_nodes = nodes[: max_nodes or total_nodes]
    allowed_node_ids = {node.id for node in limited_nodes}
    limited_edges = [
        edge for edge in edges if edge.source in allowed_node_ids and edge.target in allowed_node_ids
    ][: max_edges or total_edges]
    selected = selected_requirement.model_dump(exclude_none=True)
    if max_description_chars <= 0:
        selected.pop("description", None)
    elif len(selected.get("description", "")) > max_description_chars:
        selected["description"] = selected["description"][:max_description_chars]
    impact_facts = build_impact_facts(selected_requirement, impact_graph, max_description_chars)
    coverage_flags = build_coverage_flags(impact_facts)
    return {
        "changeText": change_text,
        "selectedRequirement": selected,
        "impactFacts": impact_facts,
        "coverageFlags": coverage_flags,
        "impactedNodes": [
            {
                "id": node.id,
                "type": node.type,
                "name": node.name,
                "criticality": node.criticality,
                "hop": node.hop,
            }
            for node in limited_nodes
        ],
        "impactedRelationships": [
            {
                "source": edge.source,
                "relationship": edge.relationship,
                "target": edge.target,
            }
            for edge in limited_edges
        ],
        "contextStats": {
            "nodesSent": len(limited_nodes),
            "totalNodes": total_nodes,
            "edgesSent": len(limited_edges),
            "totalEdges": total_edges,
            "truncated": len(limited_nodes) < total_nodes or len(limited_edges) < total_edges,
        },
    }


def build_impact_facts(
    selected_requirement: SelectedRequirement,
    impact_graph: ImpactGraph,
    max_description_chars: int = 180,
) -> dict[str, Any]:
    facts: dict[str, Any] = {
        "selectedRequirement": {
            "id": selected_requirement.id,
            "name": selected_requirement.name,
            "criticality": selected_requirement.criticality,
        },
        "connectedRequirements": [],
        "connectedSubsystems": [],
        "connectedSoftwareModules": [],
        "connectedTestCases": [],
        "connectedRisks": [],
        "connectedIssues": [],
        "connectedDocuments": [],
        "connectedTeams": [],
        "relationships": [],
    }
    for node in _sorted_nodes(impact_graph.nodes, selected_requirement.id):
        if node.id == selected_requirement.id or node.status == "selected":
            continue
        category = FACT_CATEGORIES.get(node.type)
        if not category:
            continue
        facts[category].append(_fact_node(node, max_description_chars))
    for edge in _sorted_edges(impact_graph.edges):
        facts["relationships"].append(
            {
                "source": edge.source,
                "relationship": edge.relationship,
                "target": edge.target,
                "hop": edge.hop,
            }
        )
    return facts


def build_coverage_flags(impact_facts: dict[str, Any]) -> dict[str, bool]:
    return {
        "hasSubsystemImpact": bool(impact_facts["connectedSubsystems"]),
        "hasSoftwareImpact": bool(impact_facts["connectedSoftwareModules"]),
        "hasVerificationImpact": bool(impact_facts["connectedTestCases"]),
        "hasRiskImpact": bool(impact_facts["connectedRisks"]),
        "hasIssueImpact": bool(impact_facts["connectedIssues"]),
        "hasDocumentImpact": bool(impact_facts["connectedDocuments"]),
        "hasRequirementRipple": bool(impact_facts["connectedRequirements"]),
    }


def _fact_node(node, max_description_chars: int) -> dict[str, Any]:
    description = node.description or ""
    if max_description_chars <= 0:
        description = ""
    elif len(description) > max_description_chars:
        description = description[:max_description_chars]
    return {
        "id": node.id,
        "name": node.name,
        "criticality": node.criticality,
        "hop": node.hop,
        "description": description,
    }


def _sorted_nodes(nodes, selected_id: str):
    return sorted(
        nodes,
        key=lambda node: (
            0 if node.id == selected_id else 1,
            node.hop if node.hop is not None else 99,
            CRITICALITY_PRIORITY.get(node.criticality, 9),
            TYPE_PRIORITY.get(node.type, 99),
            node.id,
        ),
    )


def _sorted_edges(edges):
    return sorted(edges, key=lambda edge: (edge.hop if edge.hop is not None else 99, edge.source, edge.relationship, edge.target))


def _log_prompt_diagnostics(context: dict[str, Any]) -> None:
    prompt = json.dumps(context, ensure_ascii=True)
    stats = context["contextStats"]
    logger.info("[LLM][Ollama] nodes sent: %s", stats["nodesSent"])
    logger.info("[LLM][Ollama] edges sent: %s", stats["edgesSent"])
    logger.info("[LLM][Ollama] prompt chars: %s", len(prompt))
    logger.info("[LLM][Ollama] approximate tokens: %s", max(1, len(prompt) // 4))


def _parse_analysis(
    content: str | None,
    provider: str,
    input: LLMImpactAnalysisInput | None = None,
) -> LLMImpactAnalysisResult:
    if not content:
        raise ImpactAnalysisFailed(f"{provider} returned an empty impact analysis response.")
    try:
        analysis = LLMImpactAnalysisResult.model_validate_json(content)
    except (ValidationError, ValueError, TypeError) as exc:
        if provider == "ollama" and input is not None:
            try:
                return _coerce_ollama_json(content, input)
            except Exception:
                pass
        logger.exception("%s impact analysis JSON parse failed: %s", provider, _safe_error_message(exc))
        raise ImpactAnalysisFailed(f"{provider} returned invalid JSON: {_safe_error_message(exc)}") from exc
    return analysis


def _coerce_ollama_json(content: str, input: LLMImpactAnalysisInput) -> LLMImpactAnalysisResult:
    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise ValueError("Ollama JSON response was not an object.")
    nested = payload.get("impactAnalysis")
    source = nested if isinstance(nested, dict) else payload
    impacted_nodes = [node for node in input.impactGraph.nodes if node.status != "selected"]
    affected_ids = [node.id for node in impacted_nodes]
    summary = _first_string(
        source.get("summary"),
        source.get("analysis"),
        source.get("explanation"),
        payload.get("summary"),
    ) or (
        f"AI reviewed {input.selectedRequirement.id} - {input.selectedRequirement.name} "
        f"against {len(impacted_nodes)} impacted graph node(s)."
    )
    ripple_effects = _ripple_effects(source.get("rippleEffects"))
    if not ripple_effects:
        ripple_effects = _graph_derived_ripple_effects(input.impactGraph)
    return LLMImpactAnalysisResult(
        provider="ollama",
        summary=summary,
        rippleEffects=ripple_effects,
        suggestedNextSteps=_string_list(source.get("suggestedNextSteps"))
        or _string_list(source.get("nextSteps"))
        or ["Review the generated impact summary with the responsible engineer."],
        engineeringReviewChecklist=_string_list(source.get("engineeringReviewChecklist"))
        or _string_list(source.get("reviewChecklist"))
        or ["Confirm the selected requirement and impacted graph nodes are correct."],
        assumptionsAndLimitations=_string_list(source.get("assumptionsAndLimitations"))
        or ["The generated JSON was normalized by the backend to match the UI schema."],
        humanInTheLoopNotice=_first_string(source.get("humanInTheLoopNotice")) or HITL_NOTICE,
    )


def _first_string(*values: Any) -> str:
    return next((value.strip() for value in values if isinstance(value, str) and value.strip()), "")


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _ripple_effects(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    effects: list[dict[str, Any]] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            continue
        affected_nodes = item.get("affectedNodes")
        effects.append(
            {
                "area": _first_string(item.get("area")) or f"Impact area {index}",
                "explanation": _first_string(item.get("explanation"), item.get("summary")) or "Review this impact area.",
                "affectedNodes": affected_nodes if isinstance(affected_nodes, list) else [],
            }
        )
    return effects


def _finalize_analysis(
    analysis: LLMImpactAnalysisResult,
    impact_graph: ImpactGraph,
    provider: str,
    selected_requirement: SelectedRequirement,
) -> LLMImpactAnalysisResult:
    analysis.provider = provider
    impact_facts = build_impact_facts(selected_requirement, impact_graph)
    analysis.summary = _final_summary(analysis.summary, selected_requirement, impact_graph)
    analysis.rippleEffects = _sanitize_ripple_effects(
        analysis.rippleEffects,
        impact_graph,
    )
    analysis.suggestedNextSteps = _patch_suggested_next_steps(
        _sanitize_string_list(analysis.suggestedNextSteps),
        impact_facts,
    )
    node_ids = {node.id for node in impact_graph.nodes}
    for effect in analysis.rippleEffects:
        effect.affectedNodes = [node_id for node_id in effect.affectedNodes if node_id in node_ids]
    analysis.affectedNodeSummary = [
        summary
        for summary in analysis.affectedNodeSummary
        if summary.nodeId in node_ids
    ]
    if not analysis.assumptionsAndLimitations:
        analysis.assumptionsAndLimitations = [
            "This analysis is limited to the nodes and relationships returned by the Neo4j impact traversal."
        ]
    if not analysis.humanInTheLoopNotice.strip():
        analysis.humanInTheLoopNotice = HITL_NOTICE
    return analysis



def _without_existing_review_prefix(summary: str) -> str:
    return re.sub(
        r"^AI reviewed\s+REQ[-_A-Za-z0-9]+\s+(?:—|-)\s+.*?(?:dependencies\.|graph node\(s\)\.)\s*",
        "",
        summary,
        flags=re.IGNORECASE,
    )


def _summary_prefix(selected_requirement: SelectedRequirement, impact_graph: ImpactGraph) -> str:
    names = _connected_node_names(impact_graph, limit=3)
    if names:
        return (
            f"AI reviewed {selected_requirement.id} \u2014 {selected_requirement.name} "
            f"for impacts involving {', '.join(names)}."
        )
    return f"AI reviewed {selected_requirement.id} \u2014 {selected_requirement.name}."


def _final_summary(summary: str, selected_requirement: SelectedRequirement, impact_graph: ImpactGraph) -> str:
    prefix = _summary_prefix(selected_requirement, impact_graph)
    cleaned = _sanitize_text(summary)
    if cleaned.startswith(prefix):
        return cleaned
    second_sentence = _without_existing_review_prefix(cleaned).strip()
    if not second_sentence:
        return prefix
    return f"{prefix} {second_sentence}"


def _sanitize_ripple_effects(ripple_effects, impact_graph: ImpactGraph) -> list[Any]:
    clean_effects = []
    seen = set()
    for effect in ripple_effects:
        effect.area = _sanitize_text(effect.area)
        effect.explanation = _sanitize_text(effect.explanation)
        if not effect.area or not effect.explanation:
            continue
        key = (effect.area.lower(), effect.explanation.lower())
        if key in seen:
            continue
        seen.add(key)
        clean_effects.append(effect)
        if len(clean_effects) == MAX_RIPPLE_EFFECTS:
            break
    if clean_effects and not _all_effects_generic(clean_effects, impact_graph):
        specific_effects = [effect for effect in clean_effects if not isGenericRippleEffect(effect)]
        return specific_effects[:MAX_RIPPLE_EFFECTS] if specific_effects else clean_effects[:MAX_RIPPLE_EFFECTS]
    impacted_nodes = [node for node in impact_graph.nodes if node.status != "selected"]
    if not impacted_nodes:
        return []
    return _graph_derived_ripple_effects(impact_graph)


def isGenericRippleEffect(effect: Any) -> bool:
    area = _sanitize_text(getattr(effect, "area", "")).lower()
    explanation = _sanitize_text(getattr(effect, "explanation", "")).lower()
    return (
        not area
        or area in {"impact", "impacts", "knowledge graph", "knowledge graph impact"}
        or "knowledge graph impact" in area
        or "review the connected impacted nodes" in explanation
        or "connected impacted nodes and relationships" in explanation
    )


def _all_ripple_effects_generic(analysis: LLMImpactAnalysisResult, impact_graph: ImpactGraph) -> bool:
    return _analysis_is_generic(analysis, impact_graph)


def _analysis_is_generic(analysis: LLMImpactAnalysisResult, impact_graph: ImpactGraph) -> bool:
    if not _has_meaningful_impacted_nodes(impact_graph):
        return False
    return _is_generic_summary(analysis.summary, impact_graph) or _all_effects_generic(analysis.rippleEffects, impact_graph)


def _is_generic_summary(summary: str, impact_graph: ImpactGraph) -> bool:
    cleaned = _sanitize_text(summary).lower()
    if not cleaned:
        return True
    has_node_name = any(name.lower() in cleaned for name in _connected_node_names(impact_graph, limit=12))
    generic_phrase = (
        "connected knowledge graph dependencies" in cleaned
        or "connected graph element" in cleaned
        or "review these dependencies" in cleaned
        or "impact map" in cleaned
    )
    return generic_phrase and not has_node_name


def _all_effects_generic(ripple_effects: list[Any], impact_graph: ImpactGraph) -> bool:
    if not ripple_effects or not _has_meaningful_impacted_nodes(impact_graph):
        return False
    return all(isGenericRippleEffect(effect) for effect in ripple_effects)


def _has_meaningful_impacted_nodes(impact_graph: ImpactGraph) -> bool:
    return any(node.status != "selected" and node.type in GRAPH_SPECIFIC_TYPES for node in impact_graph.nodes)


def _graph_derived_ripple_effects(impact_graph: ImpactGraph) -> list[RippleEffect]:
    nodes_by_type: dict[str, list[Any]] = {}
    for node in _sorted_nodes(impact_graph.nodes, ""):
        if node.status == "selected":
            continue
        nodes_by_type.setdefault(node.type, []).append(node)

    effects: list[RippleEffect] = []

    def add_effect(area: str, explanation: str, types: tuple[str, ...]) -> None:
        if len(effects) == MAX_RIPPLE_EFFECTS:
            return
        node_ids = [node.id for node_type in types for node in nodes_by_type.get(node_type, [])]
        if not node_ids:
            return
        effects.append(RippleEffect(area=area, explanation=explanation, affectedNodes=node_ids[:8]))

    subsystem = nodes_by_type.get("Subsystem", [])
    if subsystem:
        first = subsystem[0]
        add_effect(
            first.name or "System / subsystem impact",
            "The selected requirement is linked to this subsystem, so engineers should review whether the change affects subsystem behavior or allocation.",
            ("Subsystem",),
        )
    software = nodes_by_type.get("SoftwareModule", [])
    if software:
        first = software[0]
        add_effect(
            first.name or "Software module impact",
            "The connected software module may need review to confirm that its behavior still satisfies the changed requirement.",
            ("SoftwareModule",),
        )
    add_effect(
        "Verification impact",
        "Linked test cases may need updates or reruns to confirm the changed requirement remains verified.",
        ("TestCase", "Test"),
    )
    add_effect(
        "Risk and issue impact",
        "Connected risks or issues should be reviewed to determine whether the change increases exposure or requires mitigation.",
        ("Risk", "Issue"),
    )
    add_effect(
        "Documentation impact",
        "Linked specifications or analysis documents may need updates after engineering approval.",
        ("Document",),
    )
    add_effect(
        "Related requirement impact",
        "Related requirements should be reviewed to confirm the change remains consistent across dependent requirements.",
        ("Requirement",),
    )

    if effects:
        return effects[:MAX_RIPPLE_EFFECTS]
    impacted_nodes = [node for node in impact_graph.nodes if node.status != "selected"]
    return [
        RippleEffect(
            area=impacted_nodes[0].name or impacted_nodes[0].id,
            explanation="The selected requirement is linked to this impacted graph element and should be reviewed before approval.",
            affectedNodes=[impacted_nodes[0].id],
        )
    ] if impacted_nodes else []


def _graph_derived_next_steps(selected_requirement: SelectedRequirement, impact_graph: ImpactGraph) -> list[str]:
    names_by_type: dict[str, list[str]] = {}
    for node in _sorted_nodes(impact_graph.nodes, selected_requirement.id):
        if node.status == "selected":
            continue
        if node.name:
            names_by_type.setdefault(node.type, []).append(node.name)
    steps = [f"Review {selected_requirement.id} with the responsible engineer and confirm the intended change."]
    if names_by_type.get("Subsystem") or names_by_type.get("SoftwareModule"):
        names = _first_names(names_by_type.get("Subsystem", []) + names_by_type.get("SoftwareModule", []), 2)
        steps.append(f"Check impacted system and software areas: {names}.")
    if names_by_type.get("TestCase") or names_by_type.get("Test"):
        names = _first_names(names_by_type.get("TestCase", []) + names_by_type.get("Test", []), 2)
        steps.append(f"Review linked verification coverage: {names}.")
    if names_by_type.get("Risk") or names_by_type.get("Issue"):
        names = _first_names(names_by_type.get("Risk", []) + names_by_type.get("Issue", []), 2)
        steps.append(f"Assess connected risks or issues: {names}.")
    if names_by_type.get("Document"):
        steps.append(f"Update linked documents or specifications: {_first_names(names_by_type['Document'], 2)}.")
    return steps[:5]


def _connected_node_names(impact_graph: ImpactGraph, limit: int = 3) -> list[str]:
    names = []
    seen = set()
    for node in _sorted_nodes(impact_graph.nodes, ""):
        name = (node.name or node.id or "").strip()
        if node.status == "selected" or not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        names.append(name)
        seen.add(key)
        if len(names) == limit:
            break
    return names


def _first_names(names: list[str], limit: int) -> str:
    return ", ".join(names[:limit])


def _sanitize_string_list(items: list[str]) -> list[str]:
    clean_items = []
    for item in items:
        cleaned = _sanitize_text(item)
        if cleaned:
            clean_items.append(cleaned)
    return clean_items


def _sanitize_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s{2,}", " ", PROVIDER_WORD_PATTERN.sub("AI", str(value))).strip()


def _combined_public_text(summary: str, ripple_effects, suggested_next_steps: list[str]) -> str:
    parts = [summary, *[effect.area for effect in ripple_effects], *[effect.explanation for effect in ripple_effects], *suggested_next_steps]
    return " ".join(parts).lower()


def _patch_suggested_next_steps(suggested_next_steps: list[str], impact_facts: dict[str, Any]) -> list[str]:
    steps = [step for step in suggested_next_steps if step]
    text = " ".join(steps).lower()
    priorities = [
        (
            bool(impact_facts["selectedRequirement"]["id"]),
            ("requirement", impact_facts["selectedRequirement"]["id"].lower()),
            "Review the selected requirement and confirm the intended change.",
        ),
        (
            bool(impact_facts["connectedSubsystems"] or impact_facts["connectedSoftwareModules"]),
            ("subsystem", "software", "module"),
            f"Review linked system and software areas: {_fact_names(impact_facts['connectedSubsystems'] + impact_facts['connectedSoftwareModules'])}.",
        ),
        (
            bool(impact_facts["connectedTestCases"]),
            ("test", "verification"),
            f"Review linked verification test cases: {_fact_names(impact_facts['connectedTestCases'])}.",
        ),
        (
            bool(impact_facts["connectedRisks"] or impact_facts["connectedIssues"]),
            ("risk", "issue", "anomaly"),
            f"Review linked risks and issues: {_fact_names(impact_facts['connectedRisks'] + impact_facts['connectedIssues'])}.",
        ),
        (
            bool(impact_facts["connectedDocuments"]),
            ("document", "specification"),
            f"Review linked documents and specifications: {_fact_names(impact_facts['connectedDocuments'])}.",
        ),
    ]
    for present, keywords, patch in priorities:
        if present and not any(keyword in text for keyword in keywords):
            steps.append(patch)
            text = " ".join(steps).lower()
    return _prioritize_steps(steps, impact_facts)[:5]


def _prioritize_steps(steps: list[str], impact_facts: dict[str, Any]) -> list[str]:
    unique_steps = []
    seen = set()
    for step in steps:
        key = step.lower()
        if key not in seen:
            unique_steps.append(step)
            seen.add(key)

    def rank(step: str) -> int:
        lower = step.lower()
        if "requirement" in lower or impact_facts["selectedRequirement"]["id"].lower() in lower:
            return 0
        if any(word in lower for word in ("subsystem", "software", "module")):
            return 1
        if any(word in lower for word in ("verification", "test")):
            return 2
        if any(word in lower for word in ("risk", "issue", "anomaly")):
            return 3
        if any(word in lower for word in ("document", "specification")):
            return 4
        return 5

    return sorted(unique_steps, key=rank)


def _fact_names(items: list[dict[str, Any]], limit: int = 2) -> str:
    names = [item.get("name") or item.get("id") for item in items if item.get("name") or item.get("id")]
    return ", ".join(names[:limit]) if names else "the connected graph nodes"


def _finalize_ollama_analysis(
    analysis: LLMImpactAnalysisResult,
    input: LLMImpactAnalysisInput,
    context: dict[str, Any],
) -> LLMImpactAnalysisResult:
    stats = context.get("contextStats", {})
    if stats.get("truncated"):
        analysis.assumptionsAndLimitations.append(
            "The AI analysis used a compacted graph context containing "
            f"{stats.get('nodesSent')} of {stats.get('totalNodes')} nodes and "
            f"{stats.get('edgesSent')} of {stats.get('totalEdges')} edges."
        )
    logger.info("[LLM][Ollama] JSON parse completed")
    return _finalize_analysis(analysis, input.impactGraph, "ollama", input.selectedRequirement)


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _safe_error_message(exc: Exception) -> str:
    message = str(exc) or exc.__class__.__name__
    message = re.sub(r"sk-[A-Za-z0-9_-]+", "sk-REDACTED", message)
    return message.replace("\n", " ")[:1200]
