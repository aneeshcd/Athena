from __future__ import annotations

from io import BytesIO

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.lib import colors

from app.models import ImpactAnalysisResponse


def build_pdf_report(analysis: ImpactAnalysisResponse) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, title="Athena SE Impact Analysis")
    styles = getSampleStyleSheet()
    story = [
        Paragraph("Athena SE Impact Analysis", styles["Title"]),
        Paragraph(f"Changed node: {analysis.changed_node_label} ({analysis.changed_node_id})", styles["Normal"]),
        Spacer(1, 12),
        Paragraph("Summary", styles["Heading2"]),
        Paragraph(_escape_newlines(analysis.summary), styles["BodyText"]),
        Spacer(1, 12),
        Paragraph("Metrics", styles["Heading2"]),
    ]

    metrics = analysis.metrics
    table = Table(
        [
            ["Required Man Hours", metrics.required_man_hours],
            ["Cost Impact", f"EUR {metrics.cost_impact:,.2f}"],
            ["Engineers Affected", metrics.engineers_affected],
            ["Teams Affected", metrics.teams_affected],
            ["Project Delay Days", metrics.project_delay_days],
            ["Risk Category", metrics.risk_category],
            ["Safety Impact", metrics.safety_impact],
            ["AI Confidence", f"{metrics.ai_confidence_level:.2f}"],
        ],
        colWidths=[220, 220],
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef8")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#9aa7b6")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.extend([table, Spacer(1, 12), Paragraph("Reasoning Paths", styles["Heading2"])])

    for path in analysis.reasoning_paths:
        story.append(Paragraph(f"- {path}", styles["BodyText"]))

    story.append(Spacer(1, 12))
    story.append(Paragraph("Human-in-the-Loop Next Steps", styles["Heading2"]))
    for step in analysis.next_steps:
        story.append(Paragraph(f"- {step}", styles["BodyText"]))

    story.append(Spacer(1, 12))
    story.append(Paragraph("Source References", styles["Heading2"]))
    for reference in analysis.source_references:
        story.append(Paragraph(f"- {reference}", styles["BodyText"]))

    doc.build(story)
    return buffer.getvalue()


def _escape_newlines(text: str) -> str:
    return text.replace("\n", "<br/>")
