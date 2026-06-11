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
  const [file, setFile] = useState(null);
  const [changeText, setChangeText] = useState("");
  const [selectedRequirement, setSelectedRequirement] = useState(null);
  const [selectedElement, setSelectedElement] = useState(null);
  const [ingestSummary, setIngestSummary] = useState(null);
  const [ontology, setOntology] = useState([]);
  const [status, setStatus] = useState("Ready");
  const [isBusy, setIsBusy] = useState(false);

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
      setIngestSummary(summary);
      setGraph(initialGraph);
      setSelectedRequirement(null);
      setSelectedElement(null);
      setOntology(await fetchOntology());
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
    setIsBusy(true);
    setStatus("Building impact map from the best matching Requirement...");
    try {
      const response = await fetch(`${API_BASE_URL}/api/graph/impact-analysis`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ changeText, depth: 2 }),
      });
      if (!response.ok) throw new Error(await response.text());
      const payload = await response.json();
      setSelectedRequirement(payload.selectedRequirement);
      setGraph(payload.impactGraph);
      setSelectedElement(null);
      setStatus(`Impact map loaded from ${payload.selectedRequirement.id}.`);
    } catch (error) {
      setStatus(`Impact analysis failed: ${error.message}`);
    } finally {
      setIsBusy(false);
    }
  }

  async function fetchOntology() {
    const response = await fetch(`${API_BASE_URL}/api/graph/ontology`);
    if (!response.ok) return [];
    return response.json();
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
          <ImpactGraph graph={graph} onSelect={setSelectedElement} />
          <GraphInspector element={selectedElement} />
        </section>

        <aside className="right-panel">
          <div className="summary-header">
            <div className="section-title">
              <ShieldCheck size={18} />
              <h2>Impacted Nodes & Relationships</h2>
            </div>
          </div>
          <ImpactDetails selectedRequirement={selectedRequirement} graph={graph} />
          <OntologyPanel ontology={ontology} ingestSummary={ingestSummary} />
        </aside>
      </section>
    </main>
  );
}

function ImpactGraph({ graph, onSelect }) {
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
        <strong>No impact graph yet</strong>
        <span>Ingest an Excel artefact, enter a change, and find impacted nodes.</span>
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

function ImpactDetails({ selectedRequirement, graph }) {
  if (!selectedRequirement) {
    return (
      <div className="empty-summary compact">
        <AlertTriangle size={26} />
        <p>The selected starting requirement and impacted graph details appear after analysis.</p>
      </div>
    );
  }
  const impactedNodes = graph.nodes.filter((node) => node.status === "impacted");
  const selectedNode = graph.nodes.find((node) => node.id === selectedRequirement.id);
  return (
    <div className="impact-details">
      <div className="selected-requirement">
        <span>Selected Requirement</span>
        <strong>{selectedRequirement.id} - {selectedRequirement.name}</strong>
        <small>{selectedRequirement.criticality || "No criticality"} | {selectedNode?.type || selectedRequirement.type}</small>
      </div>

      <div className="impact-section">
        <h3>Impacted Nodes</h3>
        <div className="impact-list">
          {impactedNodes.length ? impactedNodes.map((node) => (
            <div className="impact-card" key={node.id}>
              <strong>{node.id} - {node.name}</strong>
              <span>{node.type} | {node.criticality || "No criticality"}</span>
            </div>
          )) : <span className="impact-empty">No impacted nodes were returned.</span>}
        </div>
      </div>

      <div className="impact-section">
        <h3>Connected Relationships</h3>
        <div className="impact-list">
          {graph.edges.length ? graph.edges.map((edge) => (
            <div className="impact-card" key={edge.id}>
              <strong>{edge.relationship}</strong>
              <span>{edge.source} - {edge.target}</span>
            </div>
          )) : <span className="impact-empty">No ontology relationships were returned.</span>}
        </div>
      </div>
    </div>
  );
}

function OntologyPanel({ ontology, ingestSummary }) {
  return (
    <div className="ontology-panel">
      {ingestSummary && (
        <div className={ingestSummary.invalid_edges.length ? "validation-box warning" : "validation-box"}>
          <strong>{ingestSummary.invalid_edges.length ? "Invalid ontology edges" : "Ontology validation passed"}</strong>
          <span>{ingestSummary.invalid_edges.length} invalid relationship(s) found during ingest.</span>
        </div>
      )}
      <h3>Ontology Rules</h3>
      <div className="ontology-list">
        {ontology.length ? ontology.map((rule) => (
          <span key={`${rule.source_entity}-${rule.relationship}-${rule.target_entity}`}>
            {rule.source_entity} - {rule.relationship} - {rule.target_entity}
          </span>
        )) : <span>No ontology rules loaded.</span>}
      </div>
    </div>
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
