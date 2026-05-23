import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import cytoscape from "cytoscape";
import {
  Activity,
  AlertTriangle,
  Brain,
  Download,
  FileUp,
  Network,
  Play,
  ShieldCheck,
} from "lucide-react";
import "./styles.css";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const initialGraph = { nodes: [], edges: [] };

const emptyMetrics = {
  required_man_hours: 0,
  cost_impact: 0,
  engineers_affected: 0,
  teams_affected: 0,
  project_delay_days: 0,
  risk_category: "Low",
  safety_impact: "None",
  ai_confidence_level: 0,
};

function App() {
  const [graph, setGraph] = useState(initialGraph);
  const [file, setFile] = useState(null);
  const [changeRequest, setChangeRequest] = useState("");
  const [analysis, setAnalysis] = useState(null);
  const [selectedElement, setSelectedElement] = useState(null);
  const [status, setStatus] = useState("Ready");
  const [isBusy, setIsBusy] = useState(false);
  const hasActiveGraph = graph.nodes.length > 0;

  useEffect(() => {
    fetch(`${API_BASE_URL}/api/graph`)
      .then((response) => (response.ok ? response.json() : null))
      .then((payload) => {
        if (payload?.nodes?.length) {
          setGraph(payload);
        }
      })
      .catch(() => undefined);
  }, []);

  async function uploadArtifact() {
    if (!file) {
      setStatus("Select a requirement artifact first.");
      return;
    }
    setIsBusy(true);
    setStatus("Ingesting artifact and building graph...");
    try {
      const formData = new FormData();
      formData.append("file", file);
      const response = await fetch(`${API_BASE_URL}/api/ingest`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) throw new Error(await response.text());
      const payload = await response.json();
      setGraph(payload.graph);
      setAnalysis(null);
      const firstRequirement = payload.artifact.entities.find((entity) => entity.type === "REQUIREMENT");
      if (firstRequirement) {
        setChangeRequest(
          `Change requirement ${firstRequirement.id}: ${firstRequirement.description} Analyze related requirements, similarity links, risks, teams, effort, cost, and schedule delay.`,
        );
      }
      setStatus(
        `Loaded ${payload.graph.nodes.length} nodes and ${payload.graph.edges.length} relationships from ${payload.artifact.filename}.`,
      );
    } catch (error) {
      setStatus(`Ingestion failed: ${error.message}`);
    } finally {
      setIsBusy(false);
    }
  }

  async function analyzeChange() {
    if (!hasActiveGraph) {
      setStatus("Upload a file and click Ingest Artifact before analysis.");
      return;
    }
    if (!changeRequest.trim()) {
      setStatus("Enter the requirement change before analysis.");
      return;
    }
    setIsBusy(true);
    setStatus("Tracing blast radius through the knowledge graph...");
    try {
      const response = await fetch(`${API_BASE_URL}/api/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ change_request: changeRequest }),
      });
      if (!response.ok) throw new Error(await response.text());
      const payload = await response.json();
      setAnalysis(payload);
      setGraph(payload.graph.nodes.length ? payload.graph : graph);
      setStatus(`Analysis complete for ${payload.changed_node_id}.`);
    } catch (error) {
      setStatus(`Analysis failed: ${error.message}`);
    } finally {
      setIsBusy(false);
    }
  }

  async function downloadPdf() {
    if (!analysis) {
      setStatus("Run an impact analysis before downloading a report.");
      return;
    }
    setStatus("Generating PDF report...");
    const response = await fetch(`${API_BASE_URL}/api/report/pdf`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ analysis }),
    });
    if (!response.ok) {
      setStatus("PDF generation failed.");
      return;
    }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "athena-se-impact-report.pdf";
    anchor.click();
    URL.revokeObjectURL(url);
    setStatus("PDF report downloaded.");
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <h1>Athena SE</h1>
          <p>Graph-grounded change intelligence for systems engineering.</p>
        </div>
        <div className="status-pill">
          <Activity size={16} />
          <span>{status}</span>
        </div>
      </header>

      <section className="workspace">
        <aside className="left-panel">
          <section className="panel-block">
            <div className="section-title">
              <FileUp size={18} />
              <h2>Input & Control</h2>
            </div>
            <label className="upload-zone">
              <input
                type="file"
                accept=".pdf,.txt,.md,.csv,.tsv,.xlsx,.xlsm"
                onChange={(event) => setFile(event.target.files?.[0] || null)}
              />
              <span>{file ? file.name : "Upload requirements, tests, risks, or interface specs"}</span>
            </label>
            <button className="secondary-button" onClick={uploadArtifact} disabled={isBusy}>
              <FileUp size={17} />
              Ingest Artifact
            </button>
          </section>

          <section className="panel-block fill">
            <div className="section-title">
              <Brain size={18} />
              <h2>Change Request</h2>
            </div>
            <textarea
              value={changeRequest}
              onChange={(event) => setChangeRequest(event.target.value)}
              placeholder="Upload a requirement artifact, then specify the exact requirement or behavior change..."
            />
            <button className="primary-button" onClick={analyzeChange} disabled={isBusy}>
              <Play size={18} />
              Analyze Change
            </button>
            {!hasActiveGraph && (
              <p className="control-hint">Analysis unlocks after a file is ingested.</p>
            )}
          </section>
        </aside>

        <section className="middle-panel">
          <div className="visual-header">
            <div className="section-title">
              <Network size={18} />
              <h2>Semantic Impact Map</h2>
            </div>
            <Legend />
          </div>
          <ImpactGraph
            graph={graph}
            changedNodeId={analysis?.changed_node_id}
            onSelect={setSelectedElement}
          />
          <GraphInspector element={selectedElement} />
        </section>

        <aside className="right-panel">
          <div className="summary-header">
            <div className="section-title">
              <ShieldCheck size={18} />
              <h2>Summary & Next Steps</h2>
            </div>
            <button className="icon-button" onClick={downloadPdf} title="Download PDF report">
              <Download size={18} />
            </button>
          </div>
          <Summary analysis={analysis} />
        </aside>
      </section>

      <MetricDashboard metrics={analysis?.metrics || emptyMetrics} />
    </main>
  );
}

function ImpactGraph({ graph, changedNodeId, onSelect }) {
  const ref = useRef(null);
  const cyRef = useRef(null);
  const hasGraph = graph.nodes.length > 0;

  const elements = useMemo(() => {
    const depths = changedNodeId ? calculateDepths(graph, changedNodeId) : new Map();
    const nodes = graph.nodes.map((node) => ({
      data: {
        id: node.id,
        label: node.id,
        title: node.label,
        type: node.type,
        sourceRef: node.source_ref,
        owner: node.owner,
        team: node.team,
        safety: node.safety_critical,
        confidence: node.confidence,
        changed: node.id === changedNodeId,
        depth: depths.has(node.id) ? depths.get(node.id) : 99,
      },
    }));
    const edges = graph.edges.map((edge) => ({
      data: {
        id: edge.id,
        source: edge.source,
        target: edge.target,
        label: edge.type,
        rationale: edge.rationale,
        confidence: edge.confidence ?? 0.6,
      },
    }));
    return [...nodes, ...edges];
  }, [graph, changedNodeId]);

  useEffect(() => {
    if (!ref.current || !hasGraph) return;
    if (cyRef.current) {
      cyRef.current.destroy();
    }
    const cy = cytoscape({
      container: ref.current,
      elements,
      layout: changedNodeId
        ? {
            name: "concentric",
            animate: true,
            padding: 70,
            minNodeSpacing: 92,
            levelWidth: () => 1,
            concentric: (node) => 10 - Math.min(node.data("depth") ?? 9, 9),
          }
        : { name: "cose", animate: true, padding: 70, nodeRepulsion: 16000, idealEdgeLength: 140 },
      style: [
        {
          selector: "node",
          style: {
            "background-color": "mapData(confidence, 0, 1, #a9b7c7, #22577a)",
            "border-color": "#ffffff",
            "border-width": 2,
            color: "#17202a",
            label: "data(label)",
            "font-size": 12,
            "font-weight": 700,
            "text-max-width": 88,
            "text-wrap": "ellipsis",
            "text-valign": "bottom",
            "text-margin-y": 10,
            "text-background-color": "#f8fafc",
            "text-background-opacity": 0.92,
            "text-background-padding": 3,
            width: 46,
            height: 46,
          },
        },
        { selector: 'node[type = "REQUIREMENT"]', style: { shape: "round-rectangle", "background-color": "#2a9d8f" } },
        { selector: 'node[type = "COMPONENT"]', style: { shape: "hexagon", "background-color": "#457b9d" } },
        { selector: 'node[type = "TEST"]', style: { shape: "diamond", "background-color": "#f4a261" } },
        { selector: 'node[type = "RISK"]', style: { shape: "triangle", "background-color": "#e76f51" } },
        { selector: "node[depth = 0]", style: { "background-color": "#111827", color: "#111827", width: 58, height: 58 } },
        { selector: "node[depth = 1]", style: { "background-color": "#e76f51" } },
        { selector: "node[depth = 2]", style: { "background-color": "#f4a261" } },
        { selector: "node[depth = 3]", style: { "background-color": "#e9c46a" } },
        { selector: "node[safety = true]", style: { "border-color": "#d62828", "border-width": 4 } },
        { selector: "node[changed = true]", style: { "border-color": "#111827", "border-width": 5, width: 52, height: 52 } },
        {
          selector: "edge",
          style: {
            width: "mapData(confidence, 0, 1, 1, 5)",
            "line-color": "#8a98a8",
            "target-arrow-color": "#8a98a8",
            "target-arrow-shape": "triangle",
            "curve-style": "bezier",
            label: "",
            "font-size": 9,
            "text-background-color": "#f8fafc",
            "text-background-opacity": 0.85,
            "text-background-padding": 2,
          },
        },
        {
          selector: "edge:selected",
          style: {
            label: "data(label)",
            width: 6,
            "line-color": "#111827",
            "target-arrow-color": "#111827",
          },
        },
        {
          selector: "node:selected",
          style: {
            "border-color": "#111827",
            "border-width": 5,
          },
        },
        {
          selector: 'edge[label = "SEMANTICALLY_SIMILAR"]',
          style: {
            "line-style": "dashed",
            "line-color": "#64748b",
            "target-arrow-color": "#64748b",
          },
        },
      ],
    });
    cyRef.current = cy;
    cy.on("tap", "node", (event) => {
      const node = event.target;
      onSelect?.({
        kind: "node",
        id: node.id(),
        title: node.data("title"),
        type: node.data("type"),
        sourceRef: node.data("sourceRef"),
        owner: node.data("owner"),
        team: node.data("team"),
        confidence: node.data("confidence"),
      });
      event.target.connectedEdges().animate({ style: { width: 5 } }, { duration: 180 }).delay(220).animate({ style: { width: 2 } });
    });
    cy.on("tap", "edge", (event) => {
      const edge = event.target;
      onSelect?.({
        kind: "relationship",
        id: edge.id(),
        type: edge.data("label"),
        source: edge.data("source"),
        target: edge.data("target"),
        rationale: edge.data("rationale"),
        confidence: edge.data("confidence"),
      });
    });
    cy.on("tap", (event) => {
      if (event.target === cy) onSelect?.(null);
    });
    return () => cy.destroy();
  }, [elements, changedNodeId, onSelect, hasGraph]);

  if (!hasGraph) {
    return (
      <div className="graph-empty">
        <Network size={30} />
        <strong>No uploaded graph yet</strong>
        <span>Select your requirements file and click Ingest Artifact.</span>
      </div>
    );
  }

  return <div className="graph-canvas" ref={ref} />;
}

function GraphInspector({ element }) {
  if (!element) {
    return (
      <div className="graph-inspector">
        <strong>Graph detail</strong>
        <span>Select a node or relationship to inspect its source, owner, and confidence.</span>
      </div>
    );
  }

  return (
    <div className="graph-inspector">
      <strong>{element.kind === "node" ? element.id : element.type}</strong>
      {element.kind === "node" ? (
        <>
          <span>{element.title}</span>
          <small>{element.type} | {element.team || "No team"} | {element.sourceRef || "No source"}</small>
        </>
      ) : (
        <>
          <span>{element.source} -> {element.target}</span>
          <small>{element.rationale || "No relationship rationale"} | confidence {Math.round((element.confidence || 0) * 100)}%</small>
        </>
      )}
    </div>
  );
}

function calculateDepths(graph, changedNodeId) {
  const adjacency = new Map();
  graph.edges.forEach((edge) => {
    if (!adjacency.has(edge.source)) adjacency.set(edge.source, []);
    if (!adjacency.has(edge.target)) adjacency.set(edge.target, []);
    adjacency.get(edge.source).push(edge.target);
    adjacency.get(edge.target).push(edge.source);
  });

  const depths = new Map([[changedNodeId, 0]]);
  const queue = [changedNodeId];
  while (queue.length) {
    const current = queue.shift();
    const nextDepth = depths.get(current) + 1;
    for (const next of adjacency.get(current) || []) {
      if (!depths.has(next)) {
        depths.set(next, nextDepth);
        queue.push(next);
      }
    }
  }
  return depths;
}

function Legend() {
  return (
    <div className="legend">
      <span><i className="req" />Requirement</span>
      <span><i className="comp" />Component</span>
      <span><i className="test" />Test</span>
      <span><i className="risk" />Risk</span>
      <span><i className="depth1" />Near impact</span>
    </div>
  );
}

function Summary({ analysis }) {
  if (!analysis) {
    return (
      <div className="empty-summary">
        <AlertTriangle size={28} />
        <p>Upload an artifact or use the demo graph, then run an analysis to populate reasoning paths, references, confidence, and HITL actions.</p>
      </div>
    );
  }

  return (
    <div className="summary-scroll">
      <div className="confidence-row">
        <span>Confidence</span>
        <strong>{Math.round(analysis.confidence_score * 100)}%</strong>
      </div>
      <p className="summary-text">{analysis.summary}</p>

      <h3>Reasoning Paths</h3>
      <ul>
        {analysis.reasoning_paths.map((path) => (
          <li key={path}>{path}</li>
        ))}
      </ul>

      <h3>Source References</h3>
      <ul>
        {analysis.source_references.map((reference) => (
          <li key={reference}>{reference}</li>
        ))}
      </ul>

      <h3>Findings</h3>
      <ul>
        {analysis.findings.map((finding) => (
          <li key={finding.title}>
            <strong>{finding.title}:</strong> {finding.evidence.join("; ")}
          </li>
        ))}
      </ul>

      <h3>Next Steps</h3>
      <ol>
        {analysis.next_steps.map((step) => (
          <li key={step}>{step}</li>
        ))}
      </ol>
    </div>
  );
}

function MetricDashboard({ metrics }) {
  const items = [
    ["Required Man Hours", metrics.required_man_hours],
    ["Cost Impact", `$${Number(metrics.cost_impact).toLocaleString()}`],
    ["Engineers Affected", metrics.engineers_affected],
    ["Teams Affected", metrics.teams_affected],
    ["Project Delay Estimated", `${metrics.project_delay_days} d`],
    ["Risk Category", metrics.risk_category],
    ["Safety Impact", metrics.safety_impact],
    ["AI Confidence Level", `${Math.round(metrics.ai_confidence_level * 100)}%`],
  ];

  return (
    <section className="metrics-panel">
      {items.map(([label, value]) => (
        <div className="metric-tile" key={label}>
          <span>{label}</span>
          <strong>{value}</strong>
        </div>
      ))}
    </section>
  );
}

createRoot(document.getElementById("root")).render(<App />);
