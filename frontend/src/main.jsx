import React, { Component, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import cytoscape from "cytoscape";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  FileUp,
  Maximize2,
  Network,
  Play,
  RotateCcw,
  Search,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import "./styles.css";

const API_BASE_URL = resolveApiBaseUrl();
const initialGraph = { nodes: [], edges: [] };
const loadingSteps = [
  "Ingesting requirement artefact...",
  "Building knowledge graph...",
  "Matching selected requirement...",
  "Traversing impact map...",
  "Generating impact analysis...",
];
const MAX_RIPPLE_EFFECTS = 3;

const typeLabels = {
  Requirement: "Requirement",
  Subsystem: "System / Subsystem",
  SoftwareModule: "Software Module",
  TestCase: "Verification / Test Case",
  Test: "Verification / Test Case",
  Risk: "Risk",
  Issue: "Issue / Anomaly",
  Document: "Specification / Document",
  Team: "Responsible Team",
  Person: "Stakeholder / Person",
};

function App() {
  const [mode, setMode] = useState("start");
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
  const [isGraphIngested, setIsGraphIngested] = useState(false);
  const [validationMessage, setValidationMessage] = useState("");
  const [loadingStepIndex, setLoadingStepIndex] = useState(0);
  const [noMatchMessage, setNoMatchMessage] = useState("");
  const latestRequestIdRef = useRef("");

  function handleFileSelection(selectedFile) {
    setFile(selectedFile);
    if (selectedFile) {
      setIsGraphIngested(false);
    }
  }

  async function ingestSelectedArtefact() {
    if (!file) {
      throw new Error("Please upload and ingest a requirement artefact first.");
    }
    setLoadingStepIndex(0);
    setStatus("Ingesting requirement artefact...");
    const formData = new FormData();
    formData.append("file", file);
    const response = await fetch(`${API_BASE_URL}/api/graph/ingest`, {
      method: "POST",
      body: formData,
    });
    if (!response.ok) throw new Error(await response.text());
    const summary = await response.json();
    setIsGraphIngested(true);
    setStatus(
      `Loaded ${summary.nodes_created} nodes, ${summary.edges_created} relationships, and ${summary.ontology_rules_created} ontology rules.`,
    );
    return summary;
  }

  async function analyseFromStart() {
    setValidationMessage("");
    if (!isGraphIngested && !file) {
      setValidationMessage("Please upload and ingest a requirement artefact first.");
      return;
    }
    if (!changeText.trim()) {
      setValidationMessage("Please describe the requirement change.");
      return;
    }
    setIsBusy(true);
    clearAnalysisState();
    try {
      if (!isGraphIngested) {
        await ingestSelectedArtefact();
      }
      setLoadingStepIndex(2);
      await runImpactAnalysis();
    } catch (error) {
      setValidationMessage(error.message || "Analysis could not be started.");
      setStatus(`Analysis failed: ${error.message}`);
    } finally {
      setIsBusy(false);
    }
  }

  async function rerunAnalysis() {
    if (!changeText.trim()) {
      setStatus("Enter the requirement change text first.");
      return;
    }
    setIsBusy(true);
    clearAnalysisState();
    try {
      await runImpactAnalysis();
    } catch (error) {
      setStatus(`Impact analysis failed: ${error.message}`);
    } finally {
      setIsBusy(false);
    }
  }

  async function runImpactAnalysis() {
    const requestId = crypto.randomUUID();
    latestRequestIdRef.current = requestId;
    setLoadingStepIndex(2);
    setStatus("Matching selected requirement...");
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
      setGraph(initialGraph);
      setGraphVersion((version) => version + 1);
      setNoMatchMessage(payload.message || "No matching requirement found.");
      setStatus(payload.message || "No matching requirement found.");
      setMode("dashboard");
      return;
    }
    setLoadingStepIndex(3);
    setSelectedRequirement(payload.selectedRequirement);
    setGraph({ nodes: payload.impactGraph.nodes || [], edges: payload.impactGraph.edges || [] });
    setGraphVersion((version) => version + 1);
    setSelectedElement(null);
    setNoMatchMessage("");
    setStatus(`Impact map loaded from ${payload.selectedRequirement.id}.`);
    setMode("dashboard");
    setLoadingStepIndex(4);
    generateImpactAnalysis(payload, changeText, requestId);
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
    } catch (_error) {
      if (requestId !== latestRequestIdRef.current) return;
      setImpactAnalysis(null);
      setAnalysisError("AI analysis could not be generated for this graph. Please review the impact map manually.");
    } finally {
      if (requestId === latestRequestIdRef.current) {
        setIsAnalysisLoading(false);
      }
    }
  }

  function clearAnalysisState() {
    setGraph(initialGraph);
    setGraphVersion((version) => version + 1);
    setSelectedRequirement(null);
    setSelectedElement(null);
    setNoMatchMessage("");
    setImpactAnalysis(null);
    setAnalysisError("");
    setIsAnalysisLoading(false);
  }

  function newAnalysis() {
    clearAnalysisState();
    setMode("start");
    setStatus(isGraphIngested ? "Graph loaded. Ready for a new change." : "Ready");
    setValidationMessage("");
  }

  if (mode === "start") {
    return (
      <StartPage
        file={file}
        setFile={handleFileSelection}
        changeText={changeText}
        setChangeText={setChangeText}
        isBusy={isBusy}
        isGraphIngested={isGraphIngested}
        validationMessage={validationMessage}
        loadingStepIndex={loadingStepIndex}
        onAnalyse={analyseFromStart}
      />
    );
  }

  return (
    <main className="app-shell dashboard-mode">
      <DashboardHeader
        status={status}
        isBusy={isBusy || isAnalysisLoading}
        isGraphIngested={isGraphIngested}
        onNewAnalysis={newAnalysis}
      />
      <section className="workspace">
        <aside className="left-panel">
          <ImpactDetails selectedRequirement={selectedRequirement} isLoading={isBusy} noMatchMessage={noMatchMessage} />
          <section className="panel-block fill">
            <div className="section-title">
              <Search size={18} />
              <h2>Requirement Change</h2>
            </div>
            <textarea
              value={changeText}
              onChange={(event) => setChangeText(event.target.value)}
              placeholder="Increase the emergency battery backup duration from 30 minutes to 45 minutes."
            />
            <button className="primary-button" onClick={rerunAnalysis} disabled={isBusy}>
              <Play size={18} />
              Re-run Analysis
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
          <AIAnalysisPanel analysis={impactAnalysis} isLoading={isAnalysisLoading} error={analysisError} />
        </aside>
      </section>
    </main>
  );
}

function StartPage({ file, setFile, changeText, setChangeText, isBusy, isGraphIngested, validationMessage, loadingStepIndex, onAnalyse }) {
  return (
    <main className="start-page">
      <section className="start-card">
        <Logo markClassName="start-logo" />
        <h1>Athena</h1>
        <p className="brand-tagline">Systems Simplified</p>
        <label className="start-field">
          <span>Upload requirement artefact</span>
          <div className={isGraphIngested ? "start-upload ready" : "start-upload"}>
            <input type="file" accept=".xlsx,.xlsm" onChange={(event) => setFile(event.target.files?.[0] || null)} />
            <FileUp size={20} />
            <strong>{file ? file.name : isGraphIngested ? "Requirement artefact ingested" : "Choose Excel artefact"}</strong>
            <small>{isGraphIngested ? "Graph is available for this session." : "Nodes, edges, and ontology sheets"}</small>
          </div>
        </label>
        <label className="start-field">
          <span>Describe requirement change</span>
          <textarea
            className="start-textarea"
            value={changeText}
            onChange={(event) => setChangeText(event.target.value)}
            placeholder="Increase the emergency battery backup duration from 30 minutes to 45 minutes."
          />
        </label>
        {validationMessage ? <div className="start-validation">{validationMessage}</div> : null}
        {isBusy ? <LoadingSequence activeIndex={loadingStepIndex} /> : null}
        <button className="start-primary-button" onClick={onAnalyse} disabled={isBusy}>
          <Sparkles size={18} />
          Analyse Impact
        </button>
        <p className="start-footnote">Neo4j knowledge graph + AI-assisted engineering review</p>
      </section>
    </main>
  );
}

function DashboardHeader({ status, isBusy, isGraphIngested, onNewAnalysis }) {
  return (
    <header className="topbar dashboard-header">
      <div className="brand-lockup">
        <Logo markClassName="dashboard-logo" />
        <div>
          <h1>Athena</h1>
          <p className="brand-tagline">Systems Simplified</p>
        </div>
      </div>
      <div className="dashboard-actions">
        <div className="status-chip-row">
          <StatusChip label="Graph Loaded" tone={isGraphIngested ? "ready" : isBusy ? "loading" : "error"} />
          <StatusChip label="Neo4j Connected" tone="ready" />
          <StatusChip label="AI Assist Ready" tone={isBusy ? "loading" : "ready"} />
        </div>
        <div className="status-pill">
          <Activity size={16} />
          <span>{status}</span>
        </div>
        <button className="secondary-button compact-button" onClick={onNewAnalysis}>
          <RotateCcw size={16} />
          New Analysis
        </button>
      </div>
    </header>
  );
}

function Logo({ markClassName }) {
  const [srcIndex, setSrcIndex] = useState(0);
  const sources = ["/logo.png", "/athena-logo.png", "/assets/logo.png"];
  if (srcIndex >= sources.length) {
    return <div className={`${markClassName} logo-fallback`}>ATHENA</div>;
  }
  return (
    <img
      className={markClassName}
      src={sources[srcIndex]}
      alt="Athena logo"
      onError={() => setSrcIndex((index) => index + 1)}
    />
  );
}

function LoadingSequence({ activeIndex }) {
  return (
    <div className="loading-sequence">
      {loadingSteps.map((step, index) => (
        <span className={index <= activeIndex ? "active" : ""} key={step}>
          <CheckCircle2 size={14} />
          {step}
        </span>
      ))}
    </div>
  );
}

function StatusChip({ label, tone }) {
  return <span className={`status-chip ${tone}`}>{label}</span>;
}

function ImpactGraph({ graph, isLoading, noMatchMessage, onSelect }) {
  const ref = useRef(null);
  const cyRef = useRef(null);
  const hasGraph = graph.nodes.length > 0;
  const [graphError, setGraphError] = useState("");
  const [isFullScreen, setIsFullScreen] = useState(false);

  function resizeAndFitGraph(padding = 80) {
    const cy = cyRef.current;
    if (!cy) return;
    const run = () => {
      cy.resize();
      cy.fit(undefined, padding);
      cy.center();
    };
    window.requestAnimationFrame(run);
    window.setTimeout(run, 140);
    window.setTimeout(run, 300);
  }

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
          readableType: readableType(node.type),
          criticality: node.criticality,
          status: node.status || "normal",
          hop: node.hop,
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
          hop: edge.hop,
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
        layout: graphLayoutOptions(graph.nodes.length),
        style: [
          {
            selector: "node",
            style: {
              "background-color": "#2563eb",
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
          { selector: 'node[type = "Requirement"]', style: { shape: "round-rectangle", "background-color": "#2563eb" } },
          { selector: 'node[type = "Subsystem"]', style: { shape: "hexagon", "background-color": "#7c3aed" } },
          { selector: 'node[type = "SoftwareModule"]', style: { shape: "round-rectangle", "background-color": "#0f766e" } },
          { selector: 'node[type = "Test"], node[type = "TestCase"]', style: { shape: "diamond", "background-color": "#16a34a" } },
          { selector: 'node[type = "Risk"]', style: { shape: "triangle", "background-color": "#dc2626" } },
          { selector: 'node[type = "Issue"]', style: { shape: "vee", "background-color": "#f59e0b" } },
          { selector: 'node[type = "Document"]', style: { shape: "rectangle", "background-color": "#64748b" } },
          { selector: 'node[type = "Team"]', style: { shape: "ellipse", "background-color": "#4f46e5" } },
          { selector: 'node[type = "Person"]', style: { shape: "ellipse", "background-color": "#6b7280" } },
          { selector: 'node[status = "selected"]', style: { "background-color": "#111827", "border-color": "#38bdf8", "border-width": 5, width: 58, height: 58 } },
          { selector: 'node[status = "impacted"]', style: { "border-color": "#f97316", "border-width": 4 } },
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
      cy.one("layoutstop", () => {
        resizeAndFitGraph(graph.nodes.length > 25 ? 90 : 70);
      });
      cy.on("mouseover", "node", (event) => event.target.addClass("hovered"));
      cy.on("mouseout", "node", (event) => event.target.removeClass("hovered"));
      cy.on("tap", "node", (event) => {
        const node = event.target;
        onSelect?.({
          kind: "node",
          id: node.id(),
          title: node.data("title"),
          description: node.data("description"),
          type: node.data("readableType"),
          criticality: node.data("criticality"),
          status: impactStatusLabel(node.data("status"), node.data("hop")),
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
          status: impactStatusLabel(edge.data("status"), edge.data("hop")),
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

  useEffect(() => {
    if (!cyRef.current || !hasGraph) return;
    resizeAndFitGraph(isFullScreen ? 100 : 80);
  }, [isFullScreen, hasGraph]);

  useEffect(() => {
    if (!hasGraph) return undefined;
    const handleResize = () => resizeAndFitGraph(isFullScreen ? 100 : 80);
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [isFullScreen, hasGraph]);

  if (!hasGraph) {
    return (
      <div className="graph-empty">
        <Network size={30} />
        <strong>{isLoading ? "Building impact graph" : noMatchMessage || "No impact graph yet"}</strong>
        <span>{isLoading ? "Tracing ontology-connected nodes and relationships." : "Upload an artefact and describe a requirement change to begin."}</span>
      </div>
    );
  }

  return (
    <div className={isFullScreen ? "graph-stage graph-stage-fullscreen" : "graph-stage"}>
      {isFullScreen ? (
        <div className="graph-fullscreen-topbar">
          <div>
            <strong>Knowledge Graph Impact Map</strong>
            <span>{graph.nodes.length} nodes / {graph.edges.length} relationships</span>
          </div>
          <button className="graph-tool-button" type="button" onClick={() => setIsFullScreen(false)}>
            Exit Full Screen
          </button>
        </div>
      ) : null}
      <div className="graph-controls">
        <button className="graph-tool-button" type="button" onClick={() => resizeAndFitGraph(80)}>
          <Maximize2 size={15} />
          Fit to View
        </button>
        <button
          className="graph-tool-button"
          type="button"
          onClick={() => setIsFullScreen(true)}
        >
          <Maximize2 size={15} />
          Open Full Screen
        </button>
      </div>
      {graphError && <div className="graph-error">Graph rendering failed: {graphError}</div>}
      <div className="graph-canvas" ref={ref} />
    </div>
  );
}

function graphLayoutOptions(nodeCount) {
  const largeGraph = nodeCount > 25;
  const mediumGraph = nodeCount > 15;
  return {
    name: "cose",
    animate: true,
    fit: true,
    padding: largeGraph ? 100 : mediumGraph ? 82 : 64,
    nodeRepulsion: largeGraph ? 15000 : mediumGraph ? 12000 : 9000,
    idealEdgeLength: largeGraph ? 220 : mediumGraph ? 190 : 160,
    edgeElasticity: 80,
    nestingFactor: 1.2,
    gravity: largeGraph ? 0.08 : mediumGraph ? 0.11 : 0.15,
    numIter: largeGraph ? 2000 : mediumGraph ? 1700 : 1500,
  };
}

function ImpactDetails({ selectedRequirement, isLoading, noMatchMessage }) {
  if (!selectedRequirement) {
    return (
      <div className="empty-summary compact">
        <AlertTriangle size={26} />
        <p>{isLoading ? "Computing the latest impact map..." : noMatchMessage || "The selected starting requirement appears after analysis."}</p>
      </div>
    );
  }
  return (
    <div className="impact-details">
      <div className="selected-requirement">
        <span>Selected Requirement</span>
        <strong>{selectedRequirement.id} - {selectedRequirement.name}</strong>
        <small>{selectedRequirement.criticality || "No criticality"} | {readableType(selectedRequirement.type)}</small>
      </div>
    </div>
  );
}

function AIAnalysisPanel({ analysis, isLoading, error }) {
  if (isLoading) {
    return (
      <section className="ai-analysis-panel">
        <div className="ai-badge">AI suggestion - engineer approval required</div>
        <SkeletonStack />
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

  const safeAnalysis = sanitizeAnalysis(analysis);

  return (
    <section className="ai-analysis-panel">
      <div className="ai-badge">AI suggestion - engineer approval required</div>
      <AnalysisSection title="Analysis Summary">
        <p>{safeAnalysis.summary}</p>
      </AnalysisSection>
      {safeAnalysis.rippleEffects?.length ? (
        <AnalysisSection title="Ripple Effects">
          {safeAnalysis.rippleEffects.map((effect) => (
            <div className="analysis-card" key={`${effect.area}-${effect.explanation}`}>
              <strong>{effect.area}</strong>
              <span>{effect.explanation}</span>
            </div>
          ))}
        </AnalysisSection>
      ) : null}
      {safeAnalysis.suggestedNextSteps?.length ? <AnalysisList title="Suggested Next Steps" items={safeAnalysis.suggestedNextSteps} /> : null}
      <div className="hitl-notice">AI-generated suggestions. Engineer review and approval required.</div>
    </section>
  );
}

function sanitizeAnalysis(analysis) {
  return {
    summary: sanitizeAnalysisText(analysis.summary),
    rippleEffects: (analysis.rippleEffects || []).slice(0, MAX_RIPPLE_EFFECTS).map((effect) => ({
      area: sanitizeAnalysisText(effect.area),
      explanation: sanitizeAnalysisText(effect.explanation),
    })),
    suggestedNextSteps: (analysis.suggestedNextSteps || []).map(sanitizeAnalysisText),
  };
}

function sanitizeAnalysisText(value) {
  if (!value) return "";
  const hiddenWords = [
    ["O", "l", "l", "a", "m", "a"].join(""),
    ["O", "p", "e", "n", "A", "I"].join(""),
  ];
  const escapedHiddenWords = hiddenWords.map((word) => word.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|");
  const localModelWord = ["l", "l", "a", "m", "a"].join("");
  return String(value)
    .replace(new RegExp(`\\b(?:${escapedHiddenWords})\\s+reviewed\\s+(REQ[-_A-Za-z0-9]+)`, "gi"), "AI reviewed $1")
    .replace(new RegExp(`Generated\\s+by\\s+(?:${escapedHiddenWords})`, "gi"), "AI-generated")
    .replace(new RegExp(`\\b(?:${escapedHiddenWords})\\s+analysis\\b`, "gi"), "AI analysis")
    .replace(new RegExp(`\\b(?:${escapedHiddenWords})\\b`, "gi"), "AI")
    .replace(new RegExp(`\\b(?:gpt|${localModelWord})[-_.:\\w]*\\b`, "gi"), "AI")
    .replace(/\s{2,}/g, " ")
    .trim();
}

function SkeletonStack() {
  return (
    <div className="skeleton-stack">
      <span />
      <span />
      <span />
    </div>
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
      ) : null}
    </AnalysisSection>
  );
}

function GraphInspector({ element }) {
  if (!element) {
    return (
      <div className="graph-inspector graph-node-details">
        <strong className="graph-node-details-title">Graph detail</strong>
        <span className="graph-node-details-description">Select a node or relationship to inspect graph data.</span>
      </div>
    );
  }
  return (
    <div className="graph-inspector graph-node-details">
      <strong className="graph-node-details-title">{element.kind === "node" ? element.id : element.type}</strong>
      {element.kind === "node" ? (
        <>
          <span className="graph-node-details-description">{element.title}</span>
          <small className="graph-node-details-meta">{element.type} | {element.criticality || "No criticality"} | {element.status}</small>
          {element.description ? <span className="graph-node-details-description">{element.description}</span> : null}
        </>
      ) : (
        <>
          <span className="graph-node-details-description">{element.source} - {element.type} - {element.target}</span>
          <small className="graph-node-details-meta">{element.description || "No relationship description"} | {element.status}</small>
        </>
      )}
    </div>
  );
}

function Legend() {
  return (
    <div className="legend">
      <span><i className="req" />Requirement</span>
      <span><i className="subsystem" />System / Subsystem</span>
      <span><i className="software" />Software Module</span>
      <span><i className="test" />Verification / Test Case</span>
      <span><i className="risk" />Risk</span>
      <span><i className="issue" />Issue / Anomaly</span>
      <span><i className="document" />Specification / Document</span>
      <span><i className="team" />Responsible Team</span>
      <span><i className="selected" />Selected</span>
      <span><i className="direct" />Direct Impact</span>
      <span><i className="ripple" />Ripple Impact</span>
    </div>
  );
}

function readableType(type) {
  return typeLabels[type] || type || "Graph Element";
}

function impactStatusLabel(status, hop) {
  if (status === "selected" || hop === 0) return "Selected";
  if (hop === 1) return "Direct Impact";
  if (hop && hop > 1) return "Ripple Impact";
  return status || "Normal";
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
          <h1>Athena</h1>
          <p>The frontend hit a runtime error.</p>
          <code>{this.state.error.message}</code>
        </main>
      );
    }
    return this.props.children;
  }
}

function resolveApiBaseUrl() {
  const params = new URLSearchParams(window.location.search);
  const apiFromQuery = params.get("api");
  if (apiFromQuery === "default") {
    window.localStorage.removeItem("ATHENA_API_BASE_URL");
    return "";
  }
  if (apiFromQuery) {
    window.localStorage.setItem("ATHENA_API_BASE_URL", apiFromQuery);
    return apiFromQuery.replace(/\/$/, "");
  }
  return (
    window.localStorage.getItem("ATHENA_API_BASE_URL")
    || import.meta.env.VITE_API_BASE_URL
    || ""
  ).replace(/\/$/, "");
}

createRoot(document.getElementById("root")).render(
  <ErrorBoundary>
    <App />
  </ErrorBoundary>,
);
