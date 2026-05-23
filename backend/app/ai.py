from __future__ import annotations

import hashlib
import json
import re
from typing import Iterable

import numpy as np
from pydantic import BaseModel

try:
    from openai import OpenAI
except ImportError:  # The deterministic parser can run without the OpenAI SDK installed.
    OpenAI = None  # type: ignore[assignment]

from app.config import Settings
from app.models import (
    EngineeringEntity,
    EngineeringRelationship,
    GraphNode,
    MetricPayload,
    NormalizedArtifact,
)


class LlmExtraction(BaseModel):
    entities: list[EngineeringEntity]
    relationships: list[EngineeringRelationship]
    warnings: list[str] = []


class ArtifactProfile(BaseModel):
    likely_format: str
    candidate_sections: list[str]
    detected_fields: list[str]
    guidance: list[str]


def get_openai_client(settings: Settings) -> OpenAI | None:
    if not settings.openai_api_key or OpenAI is None:
        return None
    return OpenAI(api_key=settings.openai_api_key)


def normalize_document(raw_text: str, filename: str, settings: Settings) -> NormalizedArtifact:
    client = get_openai_client(settings)
    document_id = _stable_id("DOC", filename + raw_text[:500])
    profile = _profile_artifact(raw_text)
    reqsim_artifact = _try_parse_reqsim_export(raw_text, filename, document_id)
    if reqsim_artifact:
        return reqsim_artifact

    if client:
        try:
            extraction = _normalize_with_openai(client, raw_text, filename, settings, profile)
            return NormalizedArtifact(
                document_id=document_id,
                filename=filename,
                entities=extraction.entities,
                relationships=extraction.relationships,
                warnings=extraction.warnings,
            )
        except Exception as exc:  # Keep ingestion deterministic if the LLM call fails.
            fallback = _generic_table_normalize(raw_text, filename, document_id) or _fallback_normalize(
                raw_text,
                filename,
                document_id,
            )
            fallback.warnings.append(f"LLM normalization unavailable: {exc}")
            return fallback

    return _generic_table_normalize(raw_text, filename, document_id) or _fallback_normalize(
        raw_text,
        filename,
        document_id,
    )


def embed_texts(texts: Iterable[str], settings: Settings) -> list[list[float]]:
    text_list = list(texts)
    client = get_openai_client(settings)
    if client:
        try:
            response = client.embeddings.create(
                model=settings.openai_embedding_model,
                input=text_list,
            )
            return [item.embedding for item in response.data]
        except Exception:
            pass

    return [_hash_embedding(text) for text in text_list]


def synthesize_summary(
    change_request: str,
    changed_node: GraphNode,
    graph_context: dict,
    metrics: MetricPayload,
    reasoning_paths: list[str],
    source_references: list[str],
    findings: list[dict],
    settings: Settings,
) -> str:
    client = get_openai_client(settings)
    if client:
        try:
            prompt = {
                "change_request": change_request,
                "changed_node": changed_node.model_dump(),
                "metrics": metrics.model_dump(),
                "reasoning_paths": reasoning_paths,
                "source_references": source_references,
                "findings": findings,
                "graph_context": graph_context,
            }
            response = client.responses.create(
                model=settings.openai_summary_model,
                input=[
                    {
                        "role": "system",
                        "content": (
                            "You are Athena, a systems engineering impact analyst. "
                            "Write a concise, source-cited impact analysis. Cite only source_ref values "
                            "present in the graph context. Include propagation path, risk, safety, and HITL actions."
                        ),
                    },
                    {"role": "user", "content": json.dumps(prompt, default=str)},
                ],
            )
            return response.output_text
        except Exception:
            pass

    references = ", ".join(source_references[:5]) or "no source references available"
    paths = "\n".join(f"- {path}" for path in reasoning_paths[:6]) or "- No downstream paths found."
    return (
        f"Change '{change_request}' maps to {changed_node.label} ({changed_node.id}). "
        f"The graph traversal identifies {len(graph_context.get('nodes', []))} impacted nodes and "
        f"{len(graph_context.get('edges', []))} relationships.\n\n"
        f"Propagation paths:\n{paths}\n\n"
        f"Estimated effort is {metrics.required_man_hours:.1f} hours with a cost impact of "
        f"{metrics.cost_impact:,.0f}. Risk is {metrics.risk_category}; safety impact is "
        f"{metrics.safety_impact}. Sources: {references}."
    )


def _normalize_with_openai(
    client: OpenAI,
    raw_text: str,
    filename: str,
    settings: Settings,
    profile: ArtifactProfile,
) -> LlmExtraction:
    response = client.responses.parse(
        model=settings.openai_extraction_model,
        input=[
            {
                "role": "system",
                "content": (
                    "Extract systems engineering knowledge from the uploaded artifact. "
                    "Engineers use inconsistent document styles, so first infer where the real requirement "
                    "content lives. Handle tables, spreadsheets, copied Word/PDF text, numbered clauses, "
                    "merged headers, repeated title rows, missing IDs, abbreviations, spelling errors, and "
                    "mixed languages as gracefully as possible. Ignore navigation text, disclaimers, revision "
                    "history, blank rows, formulas, similarity metrics, and decorative content unless they "
                    "are needed as metadata. "
                    "Return strict structured data only. Classify entities as REQUIREMENT, COMPONENT, "
                    "INTERFACE, RISK, TEST, ENGINEER, TEAM, or DOCUMENT. Extract relationships using "
                    "VALIDATES, DEPENDS_ON, CONFLICTS_WITH, DERIVED_FROM, IMPLEMENTS, OWNED_BY, "
                    "BELONGS_TO, MITIGATES, AFFECTS, or SEMANTICALLY_SIMILAR. Preserve artifact IDs, "
                    "authors, owners, teams, source references, effort, cost, delay, and safety criticality when present. "
                    "If the layout is ambiguous, still extract likely requirements with lower confidence and add a warning. "
                    "Do not invent dependencies; only create relationships supported by explicit references, shared IDs, "
                    "trace columns, validation columns, parent-child structure, or strong tabular semantics."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Filename: {filename}\n\n"
                    f"Pre-ingestion artifact profile:\n{profile.model_dump_json(indent=2)}\n\n"
                    f"Artifact text:\n{raw_text[:30000]}"
                ),
            },
        ],
        text_format=LlmExtraction,
    )
    parsed = response.output_parsed
    if parsed is None:
        raise ValueError("OpenAI returned no parsed extraction.")
    return parsed


def _fallback_normalize(raw_text: str, filename: str, document_id: str) -> NormalizedArtifact:
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    entities: list[EngineeringEntity] = [
        EngineeringEntity(
            id=document_id,
            type="DOCUMENT",
            title=filename,
            description=f"Uploaded engineering artifact {filename}",
            source_ref=filename,
            effort_hours=0,
            delay_days=0,
            confidence=0.65,
        )
    ]
    relationships: list[EngineeringRelationship] = []
    requirement_ids: list[str] = []
    test_ids: list[str] = []
    component_ids: list[str] = []

    for index, line in enumerate(lines[:120], start=1):
        lower = line.lower()
        entity_type = "REQUIREMENT"
        prefix = "REQ"
        if any(token in lower for token in ["test", "verify", "validation"]):
            entity_type = "TEST"
            prefix = "TEST"
        elif any(token in lower for token in ["component", "module", "subsystem", "service"]):
            entity_type = "COMPONENT"
            prefix = "COMP"
        elif any(token in lower for token in ["risk", "hazard", "failure", "unsafe"]):
            entity_type = "RISK"
            prefix = "RISK"
        elif any(token in lower for token in ["interface", "api", "signal", "protocol"]):
            entity_type = "INTERFACE"
            prefix = "IF"

        artifact_id = _extract_artifact_id(line) or _stable_id(prefix, f"{filename}:{index}:{line}")
        safety = any(token in lower for token in ["safety", "hazard", "critical", "brake", "medical"])
        owner = _extract_owner(line)
        team = _extract_team(line, entity_type)

        entity = EngineeringEntity(
            id=artifact_id,
            type=entity_type,  # type: ignore[arg-type]
            title=line[:90],
            description=line,
            source_ref=f"{filename}:line {index}",
            owner=owner,
            team=team,
            artifact_id=artifact_id,
            effort_hours=_estimate_effort(entity_type, safety),
            cost_rate=140 if safety else 125,
            delay_days=_estimate_delay(entity_type, safety),
            safety_critical=safety,
            confidence=0.62,
        )
        entities.append(entity)
        relationships.append(
            EngineeringRelationship(
                source_id=artifact_id,
                target_id=document_id,
                type="DERIVED_FROM",
                rationale="Entity extracted from uploaded artifact.",
                confidence=0.7,
            )
        )

        if entity_type == "REQUIREMENT":
            requirement_ids.append(artifact_id)
        elif entity_type == "TEST":
            test_ids.append(artifact_id)
        elif entity_type == "COMPONENT":
            component_ids.append(artifact_id)

    for req_id in requirement_ids:
        for comp_id in component_ids[:2]:
            relationships.append(
                EngineeringRelationship(
                    source_id=req_id,
                    target_id=comp_id,
                    type="DEPENDS_ON",
                    rationale="Fallback NLP linked requirement to nearby component.",
                    confidence=0.55,
                )
            )
        for test_id in test_ids[:2]:
            relationships.append(
                EngineeringRelationship(
                    source_id=test_id,
                    target_id=req_id,
                    type="VALIDATES",
                    rationale="Fallback NLP linked test to requirement.",
                    confidence=0.55,
                )
            )

    if not requirement_ids:
        sample_req = EngineeringEntity(
            id="REQ-SAMPLE-001",
            type="REQUIREMENT",
            title="Sample uploaded requirement",
            description=raw_text[:500] or "No readable text was found in the uploaded artifact.",
            source_ref=f"{filename}:sample",
            owner="Unassigned",
            team="Systems",
            effort_hours=12,
            delay_days=1,
            confidence=0.5,
        )
        entities.append(sample_req)
        relationships.append(
            EngineeringRelationship(
                source_id=sample_req.id,
                target_id=document_id,
                type="DERIVED_FROM",
                rationale="Fallback sample node created from unreadable artifact.",
                confidence=0.5,
            )
        )

    return NormalizedArtifact(
        document_id=document_id,
        filename=filename,
        entities=entities,
        relationships=relationships,
        warnings=["Used deterministic fallback extraction. Add OPENAI_API_KEY for LLM normalization."],
    )


def _generic_table_normalize(raw_text: str, filename: str, document_id: str) -> NormalizedArtifact | None:
    lines = [line for line in raw_text.splitlines() if line.strip()]
    entities_by_id: dict[str, EngineeringEntity] = {
        document_id: EngineeringEntity(
            id=document_id,
            type="DOCUMENT",
            title=filename,
            description="Uploaded engineering artifact with detected tabular requirement content.",
            source_ref=filename,
            effort_hours=0,
            delay_days=0,
            confidence=0.78,
            metadata={"parser": "generic-table"},
        )
    }
    relationships_by_id: dict[str, EngineeringRelationship] = {}

    current_sheet = "Sheet1"
    header: list[str] | None = None
    header_line_number = 0
    detected_rows = 0
    max_rows = 1500

    for line_number, line in enumerate(lines, start=1):
        sheet_match = re.fullmatch(r"\[sheet (.+)]", line.strip(), re.IGNORECASE)
        if sheet_match:
            current_sheet = sheet_match.group(1)
            header = None
            continue

        columns = [column.strip() for column in line.split("\t")]
        if len(columns) < 2:
            continue

        if _is_likely_requirement_header(columns):
            header = columns
            header_line_number = line_number
            continue

        if header is None:
            continue

        mapping = _map_requirement_columns(header)
        description_index = mapping.get("description")
        if description_index is None or description_index >= len(columns):
            continue

        description = columns[description_index].strip()
        if not _looks_like_requirement_text(description):
            continue

        id_index = mapping.get("id")
        explicit_id = columns[id_index].strip() if id_index is not None and id_index < len(columns) else ""
        artifact_id = _extract_artifact_id(explicit_id) or _extract_artifact_id(description)
        if not artifact_id:
            artifact_id = _stable_id("REQ", f"{filename}:{current_sheet}:{line_number}:{description}")

        type_index = mapping.get("type")
        entity_type = "REQUIREMENT"
        if type_index is not None and type_index < len(columns):
            entity_type = _entity_type_from_text(columns[type_index]) or "REQUIREMENT"

        safety = _is_safety_related(description) or _column_truthy(columns, mapping.get("safety"))
        owner = _column_value(columns, mapping.get("owner")) or _extract_owner(line)
        team = _column_value(columns, mapping.get("team")) or _team_from_source(current_sheet)
        source_ref = f"{filename}:{current_sheet}:row {line_number}"

        if artifact_id not in entities_by_id:
            entities_by_id[artifact_id] = EngineeringEntity(
                id=artifact_id,
                type=entity_type,  # type: ignore[arg-type]
                title=description[:110],
                description=description,
                source_ref=source_ref,
                owner=owner,
                team=team,
                artifact_id=artifact_id,
                effort_hours=_estimate_effort(entity_type, safety),
                cost_rate=140 if safety else 125,
                delay_days=_estimate_delay(entity_type, safety),
                safety_critical=safety,
                confidence=0.72,
                metadata={
                    "parser": "generic-table",
                    "sheet": current_sheet,
                    "header_row": header_line_number,
                },
            )

        relationships_by_id[f"{artifact_id}->DERIVED_FROM->{document_id}"] = EngineeringRelationship(
            source_id=artifact_id,
            target_id=document_id,
            type="DERIVED_FROM",
            rationale="Requirement row imported from detected tabular layout.",
            confidence=0.76,
        )

        _add_optional_linked_entities(
            columns,
            mapping,
            artifact_id,
            filename,
            current_sheet,
            line_number,
            entities_by_id,
            relationships_by_id,
        )
        detected_rows += 1
        if detected_rows >= max_rows:
            break

    if detected_rows == 0:
        return None

    return NormalizedArtifact(
        document_id=document_id,
        filename=filename,
        entities=list(entities_by_id.values()),
        relationships=list(relationships_by_id.values()),
        warnings=[
            (
                "Detected a generic tabular requirement layout. "
                f"Imported {detected_rows} requirement-like rows using header aliases; "
                "review extracted fields when source formatting is inconsistent."
            )
        ],
    )


def _try_parse_reqsim_export(raw_text: str, filename: str, document_id: str) -> NormalizedArtifact | None:
    lines = [line for line in raw_text.splitlines() if line.strip()]
    header_index = None
    for index, line in enumerate(lines):
        columns = [column.strip().lower() for column in line.split("\t")]
        if {"key1", "req1", "key2", "req2"}.issubset(set(columns)):
            header_index = index
            break

    if header_index is None:
        return None

    entities_by_id: dict[str, EngineeringEntity] = {
        document_id: EngineeringEntity(
            id=document_id,
            type="DOCUMENT",
            title=filename,
            description="Uploaded requirement similarity workbook.",
            source_ref=filename,
            effort_hours=0,
            delay_days=0,
            confidence=0.9,
            metadata={"parser": "reqsim"},
        )
    }
    relationships_by_id: dict[str, EngineeringRelationship] = {}

    parsed_rows = 0
    max_rows = 1200
    for raw_row_number, line in enumerate(lines[header_index + 1 :], start=header_index + 2):
        columns = [column.strip() for column in line.split("\t")]
        if len(columns) < 12:
            continue

        file_name_1, key_1, req_1, file_name_2, key_2, req_2 = columns[:6]
        if not (_looks_like_reqsim_key(key_1) and _looks_like_reqsim_key(key_2) and req_1 and req_2):
            continue

        parsed_rows += 1
        similarity = _best_similarity(columns[9:12])
        distance = _best_distance(columns[6:9])
        confidence = max(0.45, min(0.98, similarity or 0.65))

        for key, req_text, source_file in [
            (key_1, req_1, file_name_1),
            (key_2, req_2, file_name_2),
        ]:
            if key not in entities_by_id:
                safety = _is_safety_related(req_text)
                entities_by_id[key] = EngineeringEntity(
                    id=key,
                    type="REQUIREMENT",
                    title=req_text[:110],
                    description=req_text,
                    source_ref=f"{filename}:row {raw_row_number}:{key}",
                    owner="Unassigned",
                    team=_team_from_source(source_file),
                    artifact_id=key,
                    effort_hours=_estimate_effort("REQUIREMENT", safety),
                    cost_rate=140 if safety else 125,
                    delay_days=_estimate_delay("REQUIREMENT", safety),
                    safety_critical=safety,
                    confidence=confidence,
                    metadata={
                        "source_file": source_file,
                        "parser": "reqsim",
                    },
                )

        relationship_key = f"{key_1}->SEMANTICALLY_SIMILAR->{key_2}"
        relationships_by_id[relationship_key] = EngineeringRelationship(
            source_id=key_1,
            target_id=key_2,
            type="SEMANTICALLY_SIMILAR",
            rationale=(
                "Requirement pair imported from ReqSim workbook. "
                f"Best similarity={similarity:.3f}; best distance={distance:.3f}."
            ),
            confidence=confidence,
        )
        for key in [key_1, key_2]:
            relationships_by_id[f"{key}->DERIVED_FROM->{document_id}"] = EngineeringRelationship(
                source_id=key,
                target_id=document_id,
                type="DERIVED_FROM",
                rationale="Requirement imported from uploaded ReqSim workbook.",
                confidence=0.82,
            )

        if parsed_rows >= max_rows:
            break

    if parsed_rows == 0:
        return None

    return NormalizedArtifact(
        document_id=document_id,
        filename=filename,
        entities=list(entities_by_id.values()),
        relationships=list(relationships_by_id.values()),
        warnings=[
            (
                "Parsed ReqSim-style requirement similarity workbook deterministically. "
                f"Imported {parsed_rows} similarity rows; upload is capped to 1200 rows for interactive testing."
            )
        ],
    )


def _stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()[:8].upper()
    return f"{prefix}-{digest}"


def _hash_embedding(text: str, dimensions: int = 128) -> list[float]:
    vector = np.zeros(dimensions, dtype=float)
    for token in re.findall(r"[a-zA-Z0-9_]+", text.lower()):
        digest = hashlib.sha1(token.encode()).digest()
        index = int.from_bytes(digest[:2], "big") % dimensions
        vector[index] += 1.0
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector.tolist()


def _extract_artifact_id(line: str) -> str | None:
    dotted = re.search(r"\bG\.D\d+(?:\.\d+)+\b", line, re.IGNORECASE)
    if dotted:
        return dotted.group(0).upper()
    match = re.search(r"\b(REQ|TEST|COMP|IF|RISK|SYS|SW|HW)[-_ ]?[A-Z0-9]{2,8}\b", line, re.IGNORECASE)
    if not match:
        return None
    return match.group(0).replace(" ", "-").upper()


def _extract_owner(line: str) -> str | None:
    match = re.search(r"(owner|engineer|author)\s*[:=]\s*([A-Za-z][A-Za-z .'-]{1,50})", line, re.IGNORECASE)
    return match.group(2).strip() if match else None


def _extract_team(line: str, entity_type: str) -> str:
    match = re.search(r"team\s*[:=]\s*([A-Za-z][A-Za-z &'-]{1,50})", line, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    defaults = {
        "REQUIREMENT": "Systems",
        "COMPONENT": "Platform",
        "INTERFACE": "Integration",
        "RISK": "Safety",
        "TEST": "Verification",
    }
    return defaults.get(entity_type, "Systems")


def _estimate_effort(entity_type: str, safety: bool) -> float:
    base = {"REQUIREMENT": 12, "COMPONENT": 24, "INTERFACE": 18, "RISK": 16, "TEST": 10}.get(entity_type, 6)
    return float(base * (1.6 if safety else 1.0))


def _estimate_delay(entity_type: str, safety: bool) -> float:
    base = {"REQUIREMENT": 1, "COMPONENT": 2, "INTERFACE": 1.5, "RISK": 1.5, "TEST": 0.75}.get(entity_type, 0.25)
    return float(base * (1.8 if safety else 1.0))


def _looks_like_reqsim_key(value: str) -> bool:
    return bool(re.fullmatch(r"G\.D\d+(?:\.\d+)+", value.strip(), re.IGNORECASE))


def _best_similarity(values: list[str]) -> float:
    parsed = [_safe_float(value) for value in values]
    parsed = [value for value in parsed if value is not None]
    return max(parsed) if parsed else 0.0


def _best_distance(values: list[str]) -> float:
    parsed = [_safe_float(value) for value in values]
    parsed = [value for value in parsed if value is not None]
    return min(parsed) if parsed else 0.0


def _safe_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def _is_safety_related(text: str) -> bool:
    lower = text.lower()
    return any(
        token in lower
        for token in [
            "safety",
            "hazard",
            "collision",
            "brake",
            "warning",
            "emergency",
            "fault",
            "failure",
            "power switch",
            "signalized intersection",
        ]
    )


def _team_from_source(source_file: str) -> str:
    source = source_file.lower()
    if any(token in source for token in ["vehicle", "mosar", "tcs", "rlcs", "eirene"]):
        return "Embedded Systems"
    if any(token in source for token in ["emr", "phin", "cchit"]):
        return "Healthcare Systems"
    if any(token in source for token in ["datawarehouse", "grid", "nde"]):
        return "Data Platform"
    if any(token in source for token in ["e-store", "peppol", "kms"]):
        return "Enterprise Software"
    return "Systems"


def _profile_artifact(raw_text: str) -> ArtifactProfile:
    lines = [line for line in raw_text.splitlines() if line.strip()]
    tabular_lines = [line for line in lines if "\t" in line]
    candidate_sections: list[str] = []
    detected_fields: set[str] = set()

    for index, line in enumerate(lines[:400], start=1):
        if re.fullmatch(r"\[sheet (.+)]", line.strip(), re.IGNORECASE):
            candidate_sections.append(line.strip())
        columns = [column.strip() for column in line.split("\t")]
        if _is_likely_requirement_header(columns):
            candidate_sections.append(f"header-like row {index}: {' | '.join(columns[:10])}")
            detected_fields.update(_map_requirement_columns(columns).keys())

    likely_format = "spreadsheet/table" if len(tabular_lines) > max(3, len(lines) // 4) else "document/text"
    if any("key1" in line.lower() and "req1" in line.lower() for line in lines[:20]):
        likely_format = "requirement similarity table"

    guidance = [
        "Locate requirement IDs and requirement text before extracting metadata.",
        "Use header synonyms such as requirement, description, shall statement, object text, req id, owner, team, verification, and risk.",
        "Keep source references tied to sheet/row or page/line when possible.",
        "Lower confidence and add warnings for inferred or incomplete rows.",
    ]
    return ArtifactProfile(
        likely_format=likely_format,
        candidate_sections=candidate_sections[:12],
        detected_fields=sorted(detected_fields),
        guidance=guidance,
    )


def _is_likely_requirement_header(columns: list[str]) -> bool:
    normalized = [_normalize_header(column) for column in columns]
    has_description = any(_header_matches(column, _DESCRIPTION_HEADERS) for column in normalized)
    has_id = any(_header_matches(column, _ID_HEADERS) for column in normalized)
    has_metadata = sum(
        1
        for column in normalized
        if _header_matches(
            column,
            _OWNER_HEADERS | _TEAM_HEADERS | _TYPE_HEADERS | _TEST_HEADERS | _RISK_HEADERS | _COMPONENT_HEADERS,
        )
    )
    return has_description and (has_id or has_metadata > 0 or len(columns) >= 3)


def _map_requirement_columns(header: list[str]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    aliases = {
        "id": _ID_HEADERS,
        "description": _DESCRIPTION_HEADERS,
        "type": _TYPE_HEADERS,
        "owner": _OWNER_HEADERS,
        "team": _TEAM_HEADERS,
        "safety": _SAFETY_HEADERS,
        "test": _TEST_HEADERS,
        "component": _COMPONENT_HEADERS,
        "risk": _RISK_HEADERS,
    }
    for index, column in enumerate(header):
        normalized = _normalize_header(column)
        for field, field_aliases in aliases.items():
            if field not in mapping and _header_matches(normalized, field_aliases):
                mapping[field] = index
    return mapping


def _normalize_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _header_matches(value: str, aliases: set[str]) -> bool:
    return value in aliases or any(alias in value for alias in aliases if len(alias) >= 4)


def _looks_like_requirement_text(value: str) -> bool:
    lower = value.lower().strip()
    if len(lower) < 12:
        return False
    return any(
        token in lower
        for token in [
            "shall",
            "should",
            "must",
            "will",
            "required",
            "requirement",
            "capable of",
            "ability to",
            "provide",
            "support",
        ]
    )


def _entity_type_from_text(value: str) -> str | None:
    lower = value.lower()
    if "test" in lower or "verification" in lower:
        return "TEST"
    if "risk" in lower or "hazard" in lower:
        return "RISK"
    if "component" in lower or "module" in lower or "subsystem" in lower:
        return "COMPONENT"
    if "interface" in lower or "api" in lower or "signal" in lower:
        return "INTERFACE"
    if "requirement" in lower or "req" in lower:
        return "REQUIREMENT"
    return None


def _column_value(columns: list[str], index: int | None) -> str | None:
    if index is None or index >= len(columns):
        return None
    value = columns[index].strip()
    return value or None


def _column_truthy(columns: list[str], index: int | None) -> bool:
    value = (_column_value(columns, index) or "").lower()
    return value in {"true", "yes", "y", "1", "high", "critical", "safety critical", "asil"}


def _add_optional_linked_entities(
    columns: list[str],
    mapping: dict[str, int],
    requirement_id: str,
    filename: str,
    sheet: str,
    line_number: int,
    entities_by_id: dict[str, EngineeringEntity],
    relationships_by_id: dict[str, EngineeringRelationship],
) -> None:
    linked_specs = [
        ("component", "COMPONENT", "DEPENDS_ON", "Component referenced by requirement row."),
        ("test", "TEST", "VALIDATES", "Verification artifact referenced by requirement row."),
        ("risk", "RISK", "AFFECTS", "Risk referenced by requirement row."),
    ]
    for field, entity_type, relationship_type, rationale in linked_specs:
        value = _column_value(columns, mapping.get(field))
        if not value:
            continue
        linked_id = _extract_artifact_id(value) or _stable_id(entity_type[:4], f"{filename}:{sheet}:{line_number}:{field}:{value}")
        if linked_id not in entities_by_id:
            entities_by_id[linked_id] = EngineeringEntity(
                id=linked_id,
                type=entity_type,  # type: ignore[arg-type]
                title=value[:110],
                description=value,
                source_ref=f"{filename}:{sheet}:row {line_number}",
                team=_team_from_source(sheet),
                artifact_id=linked_id,
                effort_hours=_estimate_effort(entity_type, _is_safety_related(value)),
                delay_days=_estimate_delay(entity_type, _is_safety_related(value)),
                safety_critical=_is_safety_related(value),
                confidence=0.66,
                metadata={"parser": "generic-table", "linked_field": field},
            )
        if relationship_type == "VALIDATES":
            source_id, target_id = linked_id, requirement_id
        else:
            source_id, target_id = requirement_id, linked_id
        relationships_by_id[f"{source_id}->{relationship_type}->{target_id}"] = EngineeringRelationship(
            source_id=source_id,
            target_id=target_id,
            type=relationship_type,  # type: ignore[arg-type]
            rationale=rationale,
            confidence=0.68,
        )


_ID_HEADERS = {
    "id",
    "req id",
    "requirement id",
    "requirement number",
    "artifact id",
    "key",
    "identifier",
    "object id",
    "uid",
}
_DESCRIPTION_HEADERS = {
    "requirement",
    "requirements",
    "requirement text",
    "shall statement",
    "description",
    "text",
    "object text",
    "req",
    "requirement description",
    "system requirement",
    "user requirement",
}
_TYPE_HEADERS = {"type", "artifact type", "category", "kind", "object type"}
_OWNER_HEADERS = {"owner", "engineer", "responsible", "assignee", "author", "dr i", "dri"}
_TEAM_HEADERS = {"team", "department", "group", "discipline", "function"}
_SAFETY_HEADERS = {"safety", "safety critical", "asil", "criticality", "hazard"}
_TEST_HEADERS = {"test", "verification", "validation", "test case", "verification method", "verifies"}
_COMPONENT_HEADERS = {"component", "module", "subsystem", "system", "interface", "allocated to"}
_RISK_HEADERS = {"risk", "hazard", "failure mode", "mitigation"}
