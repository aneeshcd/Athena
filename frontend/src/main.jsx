import React, { Component, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import cytoscape from "cytoscape";
import {
  Activity,
  AlertTriangle,
  FileUp,
  Network,
  Play,
  Search,
  ShieldCheck,
} from "lucide-react";
import "./styles.css";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
const initialGraph = { nodes: [], edges: [] };

function App() {
  const [graph, setGraph] = useState(initialGraph);
  const [graphVersion, setGraphVersion] = useState(0);
  const [file, setFile] = useState(null);
  const [changeText, setChangeText] = useState("");
  const [selectedRequirement, setSelectedRequirement] = useState(null);
  const [selectedElement, setSelectedElement] = useState(null);
  const [impactAnalysis, setImpactAnalysis] = useState(null);
  const [analysisError, setAnalysisError] = useState("");
  const [isAnalysisLoading, setIsAnalysisLoading] = useState(false);
  const [status, setStatus] = useState("Ready");
  const [isBusy, setIsBusy] = useState(false);
  const [noMatchMessage, setNoMatchMessage] = useState("");
  const latestRequestIdRef = useRef("");

  async function uploadArtefact() {
    if (!file) {
      setStatus("Select an Excel artefact first.");
      return;
    }
    setIsBusy(true);
    setStatus("Ingesting Excel artefact into Neo4j...");
    try {
      const formData = new FormData();
      formData.append("file", file);
      const response = await fetch(`${API_BASE_URL}/api/graph/ingest`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) throw new Error(await response.text());
      const summary = await response.json();
      setGraph({ nodes: [], edges: [] });
      setGraphVersion((version) => version + 1);
      setSelectedRequirement(null);
      setSelectedElement(null);
      setNoMatchMessage("");
      setImpactAnalysis(null);
      setAnalysisError("");
      setIsAnalysisLoading(false);
      setStatus(
        `Loaded ${summary.nodes_created} nodes, ${summary.edges_created} relationships, and ${summary.ontology_rules_created} ontology rules.`,
      );
    } catch (error) {
      setStatus(`Ingest failed: ${error.message}`);
    } finally {
      setIsBusy(false);
    }
  }

  async function findImpactedNodes() {
    if (!changeText.trim()) {
      setStatus("Enter the requirement change text first.");
      return;
    }
    const requestId = crypto.randomUUID();
    latestRequestIdRef.current = requestId;
    setIsBusy(true);
    setGraph({ nodes: [], edges: [] });
    setGraphVersion((version) => version + 1);
    setSelectedRequirement(null);
    setSelectedElement(null);
    setNoMatchMessage("");
    setImpactAnalysis(null);
    setAnalysisError("");
    setIsAnalysisLoading(false);
    setStatus("Building impact map from the best matching Requirement...");
    try {
      const response = await fetch(`${API_BASE_URL}/api/graph/impact-from-change`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ changeText, depth: 2, requestId }),
      });
      if (!response.ok) throw new Error(await response.text());
      const payload = await response.json();
      if (payload.requestId !== latestRequestIdRef.current) return;
      if (!payload.selectedRequirement) {
        setSelectedRequirement(null);
        setGraph({ nodes: [], edges: [] });
        setGraphVersion((version) => version + 1);
        setNoMatchMessage(payload.message || "No matching requirement found.");
        setStatus(payload.message || "No matching requirement found.");
        return;
      }
      setSelectedRequirement(payload.selectedRequirement);
      setGraph({ nodes: payload.impactGraph.nodes || [], edges: payload.impactGraph.edges || [] });
      setGraphVersion((version) => version + 1);
      setSelectedElement(null);
      setNoMatchMessage("");
      setStatus(`Impact map loaded from ${payload.selectedRequirement.id}.`);
      generateImpactAnalysis(payload, changeText, requestId);
    } catch (error) {
      if (requestId !== latestRequestIdRef.current) return;
      setSelectedRequirement(null);
      setGraph({ nodes: [], edges: [] });
      setGraphVersion((version) => version + 1);
      setNoMatchMessage("No matching requirement found.");
      setImpactAnalysis(null);
      setAnalysisError("");
      setIsAnalysisLoading(false);
      setStatus(`Impact analysis failed: ${error.message}`);
    } finally {
      if (requestId === latestRequestIdRef.current) {
        setIsBusy(false);
      }
    }
  }

  async function generateImpactAnalysis(payload, submittedChangeText, requestId) {
    if (!payload.selectedRequirement || !payload.impactGraph?.nodes?.length) return;
    setIsAnalysisLoading(true);
    setAnalysisError("");
    setImpactAnalysis(null);
    try {
      const response = await fetch(`${API_BASE_URL}/api/graph/impact-analysis`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          changeText: submittedChangeText,
          selectedRequirement: payload.selectedRequirement,
          impactGraph: payload.impactGraph,
        }),
      });
      if (!response.ok) throw new Error(await response.text());
      const analysisPayload = await response.json();
      if (requestId !== latestRequestIdRef.current) return;
      setImpactAnalysis(analysisPayload.analysis);
      setAnalysisError("");
    } catch (error) {
      if (requestId !== latestRequestIdRef.current) return;
      setImpactAnalysis(null);
      setAnalysisError("AI analysis could not be generated for this graph. Please review the impact map manually.");
    } finally {
      if (requestId === latestRequestIdRef.current) {
        setIsAnalysisLoading(false);
      }
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <h1>Athena SE</h1>
          <p>Neo4j requirement change graph prototype.</p>
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
              <h2>Artefact Upload</h2>
            </div>
            <label className="upload-zone">
              <input
                type="file"
                accept=".xlsx,.xlsm"
                onChange={(event) => setFile(event.target.files?.[0] || null)}
              />
              <span>{file ? file.name : "Upload Excel with nodes, edges, and ontology sheets"}</span>
            </label>
            <button className="secondary-button" onClick={uploadArtefact} disabled={isBusy}>
              <FileUp size={17} />
              Ingest Artefact
            </button>
          </section>

          <section className="panel-block fill">
            <div className="section-title">
              <Search size={18} />
              <h2>Requirement Change</h2>
            </div>
            <textarea
              value={changeText}
              onChange={(event) => setChangeText(event.target.value)}
              placeholder="Describe the requirement change to analyze..."
            />
            <button className="primary-button" onClick={findImpactedNodes} disabled={isBusy}>
              <Play size={18} />
              Find impacted nodes
            </button>
          </section>
        </aside>

        <section className="middle-panel">
          <div className="visual-header">
            <div className="section-title">
              <Network size={18} />
              <h2>Knowledge Graph</h2>
            </div>
            <Legend />
          </div>
          <ImpactGraph key={graphVersion} graph={graph} isLoading={isBusy} noMatchMessage={noMatchMessage} onSelect={setSelectedElement} />
          <GraphInspector element={selectedElement} />
        </section>

        <aside className="right-panel">
          <div className="summary-header">
            <div className="section-title">
              <ShieldCheck size={18} />
              <h2>Impact Analysis & Suggested Next Steps</h2>
            </div>
          </div>
          <ImpactDetails selectedRequirement={selectedRequirement} isLoading={isBusy} noMatchMessage={noMatchMessage} />
          <AIAnalysisPanel analysis={impactAnalysis} isLoading={isAnalysisLoading} error={analysisError} />
        </aside>
      </section>
    </main>
  );
}

function ImpactGraph({ graph, isLoading, noMatchMessage, onSelect }) {
  const ref = useRef(null);
  const cyRef = useRef(null);
  const hasGraph = graph.nodes.length > 0;
  const [graphError, setGraphError] = useState("");

  const elements = useMemo(() => {
    const nodeIds = new Set();
    const nodes = graph.nodes.flatMap((node) => {
      const id = String(node.id || "").trim();
      if (!id || nodeIds.has(id)) return [];
      nodeIds.add(id);
      return [{
        data: {
          id,
          label: node.label || node.id,
          title: node.name || node.label || node.id,
          description: node.description,
          type: node.type,
          criticality: node.criticality,
          status: node.status || "normal",
        },
      }];
    });
    const edgeIds = new Set();
    const edges = graph.edges.flatMap((edge, index) => {
      const source = String(edge.source || "").trim();
      const target = String(edge.target || "").trim();
      if (!nodeIds.has(source) || !nodeIds.has(target)) return [];
      const relationship = edge.relationship || edge.type;
      const id = String(edge.id || `${source}-${relationship}-${target}-${index}`);
      const safeId = edgeIds.has(id) ? `${id}-${index}` : id;
      edgeIds.add(safeId);
      return [{
        data: {
          id: safeId,
          source,
          target,
          label: relationship,
          description: edge.description || edge.rationale,
          status: edge.status || "normal",
        },
      }];
    });
    return [...nodes, ...edges];
  }, [graph]);

  useEffect(() => {
    if (!ref.current || !hasGraph) return;
    if (cyRef.current) cyRef.current.destroy();
    setGraphError("");
    let cy = null;
    try {
      cy = cytoscape({
        container: ref.current,
        elements,
        layout: { name: "cose", animate: true, padding: 70, nodeRepulsion: 9000 },
        style: [
          {
            selector: "node",
            style: {
              "background-color": "#457b9d",
              "border-color": "#ffffff",
              "border-width": 2,
              color: "#17202a",
              label: "data(label)",
              "font-size": 13,
              "font-weight": 700,
              "text-max-width": 96,
              "text-wrap": "wrap",
              "text-valign": "bottom",
              "text-margin-y": 10,
              "text-background-color": "#f8fafc",
              "text-background-opacity": 0.92,
              "text-background-padding": 3,
              width: 46,
              height: 46,
            },
          },
          { selector: 'node[type = "Requirement"]', style: { shape: "round-rectangle", "background-color": "#2a9d8f" } },
          { selector: 'node[type = "Subsystem"]', style: { shape: "hexagon", "background-color": "#457b9d" } },
          { selector: 'node[type = "Test"]', style: { shape: "diamond", "background-color": "#f4a261" } },
          { selector: 'node[type = "Risk"]', style: { shape: "triangle", "background-color": "#e76f51" } },
          { selector: 'node[type = "Team"]', style: { shape: "ellipse", "background-color": "#6b7280" } },
          { selector: 'node[status = "selected"]', style: { "background-color": "#111827", "border-color": "#f4a261", "border-width": 5, width: 58, height: 58 } },
          { selector: 'node[status = "impacted"]', style: { "border-color": "#e76f51", "border-width": 4 } },
          {
            selector: "edge",
            style: {
              width: 3,
              "line-color": "#8a98a8",
              "target-arrow-color": "#8a98a8",
              "target-arrow-shape": "triangle",
              "curve-style": "bezier",
              label: "",
            },
          },
          { selector: 'edge[status = "ontology-link"]', style: { "line-color": "#e76f51", "target-arrow-color": "#e76f51", width: 4 } },
          { selector: "edge:selected", style: { label: "data(label)", width: 6, "line-color": "#111827", "target-arrow-color": "#111827" } },
          { selector: "node:selected", style: { "border-color": "#111827", "border-width": 5 } },
        ],
      });
      cyRef.current = cy;
      cy.on("tap", "node", (event) => {
        const node = event.target;
        onSelect?.({
          kind: "node",
          id: node.id(),
          title: node.data("title"),
          description: node.data("description"),
          type: node.data("type"),
          criticality: node.data("criticality"),
          status: node.data("status"),
        });
      });
      cy.on("tap", "edge", (event) => {
        const edge = event.target;
        onSelect?.({
          kind: "relationship",
          id: edge.id(),
          type: edge.data("label"),
          source: edge.data("source"),
          target: edge.data("target"),
          description: edge.data("description"),
          status: edge.data("status"),
        });
      });
      cy.on("tap", (event) => {
        if (event.target === cy) onSelect?.(null);
      });
    } catch (error) {
      setGraphError(error.message || "Graph rendering failed.");
    }
    return () => {
      if (cy) cy.destroy();
    };
  }, [elements, onSelect, hasGraph]);

  if (!hasGraph) {
    return (
      <div className="graph-empty">
        <Network size={30} />
        <strong>{isLoading ? "Building impact graph" : noMatchMessage || "No impact graph yet"}</strong>
        <span>{isLoading ? "Tracing ontology-connected nodes and relationships." : "Ingest an Excel artefact, enter a change, and find impacted nodes."}</span>
      </div>
    );
  }

  return (
    <div className="graph-stage">
      {graphError && <div className="graph-error">Graph rendering failed: {graphError}</div>}
      <div className="graph-canvas" ref={ref} />
    </div>
  );
}

function ImpactDetails({ selectedRequirement, isLoading, noMatchMessage }) {
  if (!selectedRequirement) {
    return (
      <div className="empty-summary compact">
        <AlertTriangle size={26} />
        <p>{isLoading ? "Computing the latest impact map..." : noMatchMessage || "The selected starting requirement and impacted graph details appear after analysis."}</p>
      </div>
    );
  }
  return (
    <div className="impact-details">
      <div className="selected-requirement">
        <span>Selected Requirement</span>
        <strong>{selectedRequirement.id} - {selectedRequirement.name}</strong>
        <small>{selectedRequirement.criticality || "No criticality"} | {selectedRequirement.type}</small>
      </div>
    </div>
  );
}

function AIAnalysisPanel({ analysis, isLoading, error }) {
  if (isLoading) {
    return (
      <section className="ai-analysis-panel">
        <div className="ai-badge">AI suggestion - engineer approval required</div>
        <p className="analysis-muted">Generating local AI impact analysis. First run may take longer while the model loads...</p>
      </section>
    );
  }

  if (error) {
    return (
      <section className="ai-analysis-panel warning">
        <div className="ai-badge">AI suggestion - engineer approval required</div>
        <p className="analysis-muted">{error}</p>
      </section>
    );
  }

  if (!analysis) {
    return (
      <section className="ai-analysis-panel">
        <div className="ai-badge">AI suggestion - engineer approval required</div>
        <p className="analysis-muted">AI suggestions appear after a valid impact map is generated.</p>
      </section>
    );
  }

  return (
    <section className="ai-analysis-panel">
      <div className="ai-badge">AI suggestion - engineer approval required</div>
      <AnalysisSection title="Analysis Summary">
        <p>{analysis.summary}</p>
      </AnalysisSection>
      {analysis.rippleEffects?.length ? (
        <AnalysisSection title="Ripple Effects">
          {analysis.rippleEffects.map((effect) => (
          <div className="analysis-card" key={`${effect.area}-${effect.affectedNodes?.join("-")}`}>
            <strong>{effect.area}</strong>
            <span>{effect.explanation}</span>
          </div>
          ))}
        </AnalysisSection>
      ) : null}
      {analysis.suggestedNextSteps?.length ? <AnalysisList title="Suggested Next Steps" items={analysis.suggestedNextSteps} /> : null}
      <div className="hitl-notice">AI-generated suggestions. Engineer review and approval required.</div>
    </section>
  );
}

function AnalysisSection({ title, children }) {
  return (
    <div className="analysis-section">
      <h3>{title}</h3>
      {children}
    </div>
  );
}

function AnalysisList({ title, items = [] }) {
  return (
    <AnalysisSection title={title}>
      {items.length ? (
        <ul className="analysis-list">
          {items.map((item) => <li key={item}>{item}</li>)}
        </ul>
      ) : <p className="analysis-muted">No items returned.</p>}
    </AnalysisSection>
  );
}

function GraphInspector({ element }) {
  if (!element) {
    return (
      <div className="graph-inspector">
        <strong>Graph detail</strong>
        <span>Select a node or relationship to inspect graph data.</span>
      </div>
    );
  }
  return (
    <div className="graph-inspector">
      <strong>{element.kind === "node" ? element.id : element.type}</strong>
      {element.kind === "node" ? (
        <>
          <span>{element.title}</span>
          <small>{element.type} | {element.criticality || "No criticality"} | {element.status}</small>
        </>
      ) : (
        <>
          <span>{element.source} - {element.type} - {element.target}</span>
          <small>{element.description || "No relationship description"} | {element.status}</small>
        </>
      )}
    </div>
  );
}

function Legend() {
  return (
    <div className="legend">
      <span><i className="req" />Requirement</span>
      <span><i className="comp" />Subsystem</span>
      <span><i className="test" />Test</span>
      <span><i className="risk" />Risk</span>
      <span><i className="selected" />Selected</span>
    </div>
  );
}

class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  render() {
    if (this.state.error) {
      return (
        <main className="app-error">
          <h1>Athena SE</h1>
          <p>The frontend hit a runtime error.</p>
          <code>{this.state.error.message}</code>
        </main>
      );
    }
    return this.props.children;
  }
}

createRoot(document.getElementById("root")).render(
  <ErrorBoundary>
    <App />
  </ErrorBoundary>,
);
