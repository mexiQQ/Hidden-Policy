"use client";

import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties, type PointerEvent as ReactPointerEvent } from "react";
import katex from "katex";

type Part = { type: "text" | "token"; raw: string; display: string; style?: string };
type Block = {
  id: string; kind: string; raw: string; text: string; parts: Part[]; startLine: number;
  endLine: number; page: number; editable: boolean; label: string; number?: string; statementLabel?: string;
  listType?: "ordered" | "unordered"; items?: Array<{ text: string; parts: Part[] }>;
  segments?: Array<{ type: "text"; parts: Part[] } | { type: "math"; source: string; number?: string }>;
};
type DocumentData = { title: string; branch: string; head: string; clean: boolean; pageCount: number; blocks: Block[] };
type Proposal = { updatedRaw: string; newText: string; mathSource?: string };
type DraftPreview = Proposal & { blockId: string };
type DiffPart = { type: "same" | "insert" | "delete"; value: string };
type FormulaEditing = { blockId: string; partIndex: number | null; value: string; baseRaw: string };
type StoredDraft = {
  blockId: string; originalRaw: string; kind: string; label: string; startLine: number; proposal: Proposal;
};
type StoredDraftPayload = { version: 1; head: string; savedAt: string; drafts: StoredDraft[] };

const API = "http://127.0.0.1:4318/api";
const LEGACY_DRAFT_STORAGE_KEY = "hidden-policy-review-drafts:v1";
const MATH_MACROS: Record<string, string> = {
  "\\Rint": "\\mathcal{R}", "\\Rreveal": "\\mathcal{R}^{\\prime}",
  "\\Hist": "\\mathcal{H}", "\\Act": "\\mathcal{A}", "\\Traj": "\\mathcal{T}",
  "\\HP": "\\textsc{Hidden Policy}", "\\centernot": "\\not",
};

function escapeHtml(value: string) {
  return value.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function partsFromRaw(raw: string, fallback: Part[] = []) {
  const parts: Part[] = [];
  const pattern = /(\$[^$]+\$|\\(?:citep|citet|cite|ref|eqref)\{[^}]+\}|\\(?:emph|textit|textbf|textsc)\{[^}]+\}|\\(?:HP|Rint|Rreveal|Hist|Act|Traj)\b|~|---)/g;
  let cursor = 0;
  for (const match of raw.matchAll(pattern)) {
    if (match.index > cursor) parts.push({ type: "text", raw: raw.slice(cursor, match.index), display: raw.slice(cursor, match.index) });
    const token = match[0];
    const known = fallback.find((part) => part.type === "token" && part.raw === token);
    let display = known?.display || token;
    let style = known?.style || "macro";
    let inner;
    if (token.startsWith("$")) { display = token.slice(1, -1); style = "math"; }
    else if ((inner = token.match(/^\\(?:emph|textit)\{([^}]+)\}$/))) { display = inner[1]; style = "italic"; }
    else if ((inner = token.match(/^\\textbf\{([^}]+)\}$/))) { display = inner[1]; style = "bold"; }
    else if ((inner = token.match(/^\\textsc\{([^}]+)\}$/))) { display = inner[1]; style = "smallcaps"; }
    parts.push({ type: "token", raw: token, display, style });
    cursor = match.index + token.length;
  }
  if (cursor < raw.length) parts.push({ type: "text", raw: raw.slice(cursor), display: raw.slice(cursor) });
  return parts;
}

function editorHtml(parts: Part[]) {
  return parts.map((part, index) => part.type === "text"
    ? escapeHtml(part.display)
    : `<span contenteditable="false" class="locked-token token-${escapeHtml(part.style || "macro")}" data-token="${index}" title="Formatting, formulas, and references are preserved">${escapeHtml(part.display)}</span>`
  ).join("");
}

function latexEscape(value: string) {
  return value.replace(/\\/g, "\\textbackslash{}")
    .replace(/([#$%&_{}])/g, "\\$1").replace(/\^/g, "\\textasciicircum{}")
    .replace(/~/g, "\\textasciitilde{}");
}

function serializeEditor(root: HTMLElement, parts: Part[]) {
  const walk = (node: Node): string => {
    if (node.nodeType === Node.TEXT_NODE) return latexEscape(node.textContent || "");
    if (!(node instanceof HTMLElement)) return "";
    const tokenIndex = node.dataset.token;
    if (tokenIndex !== undefined) return parts[Number(tokenIndex)]?.raw || "";
    if (node.tagName === "BR") return "\n";
    const inside = Array.from(node.childNodes).map(walk).join("");
    return node.tagName === "DIV" ? `\n${inside}` : inside;
  };
  return Array.from(root.childNodes).map(walk).join("").replace(/^\n|\n$/g, "");
}

function wordDiff(before: string, after: string): DiffPart[] {
  const a = before.match(/\s+|[^\s]+/g) || [];
  const b = after.match(/\s+|[^\s]+/g) || [];
  const table = Array.from({ length: a.length + 1 }, () => new Uint16Array(b.length + 1));
  for (let i = a.length - 1; i >= 0; i--) for (let j = b.length - 1; j >= 0; j--)
    table[i][j] = a[i] === b[j] ? table[i + 1][j + 1] + 1 : Math.max(table[i + 1][j], table[i][j + 1]);
  const result: DiffPart[] = [];
  const push = (type: DiffPart["type"], value: string) => {
    const last = result.at(-1);
    if (last?.type === type) last.value += value; else result.push({ type, value });
  };
  let i = 0, j = 0;
  while (i < a.length || j < b.length) {
    if (i < a.length && j < b.length && a[i] === b[j]) { push("same", a[i]); i++; j++; }
    else if (j < b.length && (i === a.length || table[i][j + 1] >= table[i + 1][j])) { push("insert", b[j]); j++; }
    else { push("delete", a[i]); i++; }
  }
  return result;
}

function mathSourceFromDisplay(raw: string) {
  return raw
    .replace(/^\\begin\{(?:equation|align)\*?\}\s*/, "")
    .replace(/\s*\\end\{(?:equation|align)\*?\}\s*$/, "")
    .replace(/^\\\[\s*/, "").replace(/\s*\\\]$/, "")
    .replace(/\\label\{[^}]+\}/g, "").trim();
}

function replaceDisplayMath(raw: string, source: string) {
  const begin = raw.match(/^\\begin\{(?:equation|align)\*?\}/)?.[0];
  const end = raw.match(/\\end\{(?:equation|align)\*?\}\s*$/)?.[0]?.trim();
  const labels = raw.match(/\\label\{[^}]+\}/g) || [];
  if (begin && end) return `${begin}\n${source}${labels.length ? `\n    ${labels.join("\n    ")}` : ""}\n${end}`;
  if (/^\\\[/.test(raw)) return `\\[\n${source}\n\\]`;
  return source;
}

function proposalFromFormulaEdit(block: Block, editing: FormulaEditing): Proposal {
  if (editing.partIndex === null) {
    return { updatedRaw: replaceDisplayMath(editing.baseRaw, editing.value), newText: editing.value, mathSource: editing.value };
  }
  const parts = partsFromRaw(editing.baseRaw, block.parts);
  parts[editing.partIndex] = { type: "token", raw: `$${editing.value}$`, display: editing.value, style: "math" };
  const updatedRaw = parts.map((part) => part.raw).join("");
  return { updatedRaw, newText: partsFromRaw(updatedRaw, block.parts).map((part) => part.display).join("") };
}

function draftEntries(documentData: DocumentData, proposals: Record<string, Proposal>, transient?: DraftPreview | null) {
  const merged = new Map(Object.entries(proposals));
  if (transient) {
    const block = documentData.blocks.find((item) => item.id === transient.blockId);
    if (block && transient.updatedRaw !== block.raw) merged.set(transient.blockId, transient);
    else merged.delete(transient.blockId);
  }
  return Array.from(merged, ([blockId, proposal]) => {
    const block = documentData.blocks.find((item) => item.id === blockId);
    if (!block || proposal.updatedRaw === block.raw) return null;
    return { blockId, originalRaw: block.raw, kind: block.kind, label: block.label, startLine: block.startLine, proposal } satisfies StoredDraft;
  }).filter((entry): entry is StoredDraft => Boolean(entry));
}

function proposalsFromEntries(documentData: DocumentData, entries: StoredDraft[]) {
  const restored: Record<string, Proposal> = {};
  const claimed = new Set<string>();
  for (const entry of entries) {
    if (!entry || typeof entry.originalRaw !== "string" || !entry.proposal ||
      typeof entry.proposal.updatedRaw !== "string" || typeof entry.proposal.newText !== "string") continue;
    const exact = documentData.blocks.find((block) => block.id === entry.blockId && block.raw === entry.originalRaw && !claimed.has(block.id));
    const candidates = documentData.blocks.filter((block) => block.raw === entry.originalRaw && block.kind === entry.kind && !claimed.has(block.id));
    const block = exact || candidates.sort((left, right) => Math.abs(left.startLine - entry.startLine) - Math.abs(right.startLine - entry.startLine))[0];
    if (!block || entry.proposal.updatedRaw === block.raw) continue;
    restored[block.id] = entry.proposal;
    claimed.add(block.id);
  }
  return restored;
}

function readLegacyDraftPayload() {
  try {
    const raw = window.localStorage.getItem(LEGACY_DRAFT_STORAGE_KEY);
    if (!raw) return null;
    const payload = JSON.parse(raw) as Partial<StoredDraftPayload>;
    if (payload.version !== 1 || !Array.isArray(payload.drafts)) return null;
    return payload as StoredDraftPayload;
  } catch {
    return null;
  }
}

function makeDraftPayload(documentData: DocumentData, proposals: Record<string, Proposal>, transient?: DraftPreview | null): StoredDraftPayload {
  return { version: 1, head: documentData.head, savedAt: new Date().toISOString(), drafts: draftEntries(documentData, proposals, transient) };
}

async function saveSharedDrafts(payload: StoredDraftPayload) {
  const response = await fetch(`${API}/drafts`, {
    method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(payload), keepalive: true,
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "Could not save review drafts");
}

async function readSharedDrafts(documentData: DocumentData) {
  const response = await fetch(`${API}/drafts`, { cache: "no-store" });
  const payload = await response.json() as StoredDraftPayload & { error?: string };
  if (!response.ok) throw new Error(payload.error || "Could not load review drafts");
  if (payload.version === 1 && Array.isArray(payload.drafts) && payload.drafts.length) {
    return { proposals: proposalsFromEntries(documentData, payload.drafts), migrated: false };
  }
  const legacy = readLegacyDraftPayload();
  if (legacy?.drafts.length) {
    await saveSharedDrafts({ ...legacy, head: documentData.head, savedAt: new Date().toISOString() });
    window.localStorage.removeItem(LEGACY_DRAFT_STORAGE_KEY);
    return { proposals: proposalsFromEntries(documentData, legacy.drafts), migrated: true };
  }
  return { proposals: {}, migrated: false };
}

function MathFormula({ source, displayMode = false }: { source: string; displayMode?: boolean }) {
  const html = useMemo(() => katex.renderToString(source.replace(/\\centernot\s*\\implies/g, "\\not\\Rightarrow"), {
    displayMode, throwOnError: false, strict: "ignore", trust: false, macros: MATH_MACROS,
  }), [source, displayMode]);
  return <span className={displayMode ? "math-output display-output" : "math-output inline-output"} dangerouslySetInnerHTML={{ __html: html }} />;
}

function PartsText({ parts, onEditMath }: { parts: Part[]; onEditMath?: (index: number) => void }) {
  return <>{parts.map((part, index) => part.type === "text"
    ? <span key={index}>{part.display}</span>
    : part.style === "math" && onEditMath ? <button key={index} type="button" className="inline-math-button"
      onClick={(event) => { event.stopPropagation(); onEditMath(index); }} aria-label={`Edit formula ${part.display}`} title="Click to edit formula">
      <MathFormula source={part.raw.slice(1, -1)} />
    </button>
      : part.style === "math" ? <span key={index} className="inline-math-static"><MathFormula source={part.raw.slice(1, -1)} /></span>
      : <span key={index} className={`token token-${part.style || "macro"}`}>{part.display}</span>)}</>;
}

function BlockPrefix({ block }: { block: Block }) {
  if (block.statementLabel) return <span className="statement-label">{block.statementLabel} </span>;
  if (!block.number) return null;
  return <span className={block.kind === "display" ? "equation-number" : "heading-number"}>{block.kind === "display" ? `(${block.number})` : block.number}</span>;
}

function BlockText({ block, onEditMath }: { block: Block; onEditMath: (index: number) => void }) {
  if (block.items?.length) {
    const List = block.listType === "ordered" ? "ol" : "ul";
    return <List>{block.items.map((item, index) => <li key={index}><PartsText parts={item.parts} /></li>)}</List>;
  }
  if (block.segments?.length) return <><BlockPrefix block={block} />{block.segments.map((segment, index) => segment.type === "text"
    ? <span className="statement-text" key={index}><PartsText parts={segment.parts} /></span>
    : <span className="statement-math" key={index}><MathFormula source={segment.source} displayMode />{segment.number && <span className="equation-number">({segment.number})</span>}</span>)}</>;
  return <><BlockPrefix block={block} /><PartsText parts={block.parts} onEditMath={block.editable ? onEditMath : undefined} /></>;
}

function DiffText({ before, after }: { before: string; after: string }) {
  return <>{wordDiff(before, after).map((part, index) => part.type === "same"
    ? <span key={index}>{part.value}</span>
    : part.type === "insert" ? <ins key={index}>{part.value}</ins> : <del key={index}>{part.value}</del>)}</>;
}

function FormulaDiff({ before, after }: { before: string; after: string }) {
  return <div className="formula-diff">
    <del><MathFormula source={before} displayMode /></del>
    <ins><MathFormula source={after} displayMode /></ins>
  </div>;
}

function EditorHeightHandle({ onPointerDown, onStep }: {
  onPointerDown: (event: ReactPointerEvent<HTMLButtonElement>) => void; onStep: (delta: number) => void;
}) {
  return <button type="button" className="editor-height-handle" onPointerDown={onPointerDown}
    onKeyDown={(event) => {
      if (event.key === "ArrowUp") { event.preventDefault(); onStep(-24); }
      if (event.key === "ArrowDown") { event.preventDefault(); onStep(24); }
    }} aria-label="Resize editor height" title="Drag up or down to resize"><span /></button>;
}

function FormulaEditor({ value, displayMode, height, onChange, onDone, onCancel, onResizeStart, onResizeStep }: {
  value: string; displayMode: boolean; height: number; onChange: (value: string) => void; onDone: () => void; onCancel: () => void;
  onResizeStart: (event: ReactPointerEvent<HTMLButtonElement>) => void; onResizeStep: (delta: number) => void;
}) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  useEffect(() => { textareaRef.current?.focus(); }, []);
  return <div className={`formula-editor${displayMode ? " formula-editor-display" : ""}`}>
    <div className="formula-live-preview"><MathFormula source={value} displayMode={displayMode} /></div>
    <textarea ref={textareaRef} value={value} style={{ height }} onChange={(event) => onChange(event.target.value)}
      onKeyDown={(event) => {
        if (event.key === "Escape") { event.preventDefault(); onCancel(); }
        if ((event.metaKey || event.ctrlKey) && event.key === "Enter") { event.preventDefault(); onDone(); }
      }} aria-label="Edit LaTeX formula" spellCheck={false} />
    <EditorHeightHandle onPointerDown={onResizeStart} onStep={onResizeStep} />
    <div className="formula-editor-footer"><span>Edit LaTeX · live preview</span><div>
      <button type="button" className="formula-cancel" onClick={onCancel}>Cancel</button>
      <button type="button" className="formula-done" onClick={onDone}>Track change</button>
    </div></div>
  </div>;
}

function lineNumbers(page: number) {
  const start = (page - 1) * 54;
  return Array.from({ length: 54 }, (_, index) => String(start + index).padStart(3, "0"));
}

export function PaperEditor() {
  const [document, setDocument] = useState<DocumentData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [editing, setEditing] = useState<string | null>(null);
  const [formulaEditing, setFormulaEditing] = useState<FormulaEditing | null>(null);
  const [proposals, setProposals] = useState<Record<string, Proposal>>({});
  const [editingParts, setEditingParts] = useState<Part[]>([]);
  const [draftPreview, setDraftPreview] = useState<DraftPreview | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [notice, setNotice] = useState("");
  const [panelWidth, setPanelWidth] = useState(380);
  const [editorHeight, setEditorHeight] = useState(360);
  const [draftsReady, setDraftsReady] = useState(false);
  const editorRef = useRef<HTMLDivElement>(null);

  const load = useCallback(async () => {
    setLoading(true); setError("");
    try {
      const response = await fetch(`${API}/document`, { cache: "no-store" });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Could not load the manuscript");
      setDocument(payload);
      const { proposals: restored, migrated } = await readSharedDrafts(payload);
      setProposals(restored);
      setDraftsReady(true);
      const restoredCount = Object.keys(restored).length;
      if (restoredCount) setNotice(`${migrated ? "Migrated and restored" : "Restored"} ${restoredCount} shared draft${restoredCount === 1 ? "" : "s"}.`);
    } catch (caught) { setError(caught instanceof Error ? caught.message : "Could not load the manuscript"); }
    finally { setLoading(false); }
  }, []);

  // The first client render must hydrate from the local manuscript service.
  // eslint-disable-next-line react-hooks/set-state-in-effect
  useEffect(() => { load(); }, [load]);
  useEffect(() => {
    if (!editing || !editorRef.current) return;
    const selection = window.getSelection();
    const range = window.document.createRange();
    if (selection && range) { range.selectNodeContents(editorRef.current); range.collapse(false); selection.removeAllRanges(); selection.addRange(range); }
    editorRef.current.focus();
  }, [editing]);

  const transientDraft = useMemo<DraftPreview | null>(() => {
    if (draftPreview) return draftPreview;
    if (!document || !formulaEditing) return null;
    const block = document.blocks.find((item) => item.id === formulaEditing.blockId);
    return block ? { blockId: block.id, ...proposalFromFormulaEdit(block, formulaEditing) } : null;
  }, [document, draftPreview, formulaEditing]);

  useEffect(() => {
    if (!draftsReady || !document) return;
    const payload = makeDraftPayload(document, proposals, transientDraft);
    const timer = window.setTimeout(() => {
      saveSharedDrafts(payload).catch((caught) => setError(caught instanceof Error ? caught.message : "Could not autosave review drafts"));
    }, 300);
    return () => window.clearTimeout(timer);
  }, [document, draftsReady, proposals, transientDraft]);

  const pages = useMemo(() => {
    const grouped = new Map<number, Block[]>();
    for (const block of document?.blocks || []) {
      const page = Math.min(block.page, document?.pageCount || 1);
      grouped.set(page, [...(grouped.get(page) || []), block]);
    }
    return grouped;
  }, [document]);

  function startTextEdit(block: Block) {
    const proposal = proposals[block.id];
    const updatedRaw = proposal?.updatedRaw || block.raw;
    setFormulaEditing(null);
    setEditingParts(partsFromRaw(updatedRaw, block.parts));
    setDraftPreview({ blockId: block.id, updatedRaw, newText: proposal?.newText || block.text });
    setEditing(block.id);
  }

  function updateDraftPreview(block: Block) {
    if (!editorRef.current) return;
    setDraftPreview({
      blockId: block.id,
      updatedRaw: serializeEditor(editorRef.current, editingParts),
      newText: editorRef.current.innerText.replace(/\s+/g, " ").trim(),
    });
  }

  function closeTextEditor() {
    setEditing(null);
    setEditingParts([]);
    setDraftPreview(null);
  }

  function finishEditing(block: Block) {
    if (!editorRef.current) return;
    const updatedRaw = serializeEditor(editorRef.current, editingParts);
    const newText = editorRef.current.innerText.replace(/\s+/g, " ").trim();
    if (updatedRaw !== block.raw) setProposals((current) => ({ ...current, [block.id]: { updatedRaw, newText } }));
    else setProposals((current) => { const next = { ...current }; delete next[block.id]; return next; });
    closeTextEditor();
  }

  function startFormulaEdit(block: Block, partIndex: number | null) {
    closeTextEditor();
    const baseRaw = proposals[block.id]?.updatedRaw || block.raw;
    if (partIndex === null) {
      setFormulaEditing({ blockId: block.id, partIndex, value: mathSourceFromDisplay(baseRaw), baseRaw });
      return;
    }
    const parts = proposals[block.id] ? partsFromRaw(baseRaw, block.parts) : block.parts;
    const part = parts[partIndex];
    if (!part || part.style !== "math") return;
    setFormulaEditing({ blockId: block.id, partIndex, value: part.raw.slice(1, -1), baseRaw });
  }

  function finishFormulaEdit(block: Block) {
    if (!formulaEditing || formulaEditing.blockId !== block.id) return;
    const proposal = proposalFromFormulaEdit(block, formulaEditing);
    if (proposal.updatedRaw === block.raw) setProposals((current) => { const next = { ...current }; delete next[block.id]; return next; });
    else setProposals((current) => ({ ...current, [block.id]: proposal }));
    setFormulaEditing(null);
  }

  async function accept(block: Block) {
    const proposal = proposals[block.id];
    if (!proposal) return;
    setBusy(block.id); setError(""); setNotice("");
    try {
      const response = await fetch(`${API}/accept`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ blockId: block.id, updatedRaw: proposal.updatedRaw }) });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Could not accept the change");
      setDocument(payload.document);
      setProposals((current) => {
        if (!document) return {};
        const remaining = draftEntries(document, current).filter((entry) => entry.blockId !== block.id);
        return proposalsFromEntries(payload.document, remaining);
      });
      if (editing === block.id) closeTextEditor();
      if (formulaEditing?.blockId === block.id) setFormulaEditing(null);
      setNotice(`Accepted and committed as ${payload.commit}`);
    } catch (caught) { setError(caught instanceof Error ? caught.message : "Could not accept the change"); }
    finally { setBusy(null); }
  }

  function reject(block: Block) {
    setProposals((current) => { const next = { ...current }; delete next[block.id]; return next; });
    closeTextEditor(); setFormulaEditing(null); setNotice("Change rejected; original text restored.");
  }

  function blockClass(block: Block) { return `paper-block kind-${block.kind}${block.editable ? " can-edit" : ""}`; }

  function clampPanelWidth(width: number) {
    return Math.max(310, Math.min(width, Math.min(720, Math.max(330, window.innerWidth - 420))));
  }

  function startPanelResize(event: ReactPointerEvent<HTMLDivElement>) {
    event.preventDefault();
    const startX = event.clientX;
    const startWidth = panelWidth;
    const move = (moveEvent: PointerEvent) => setPanelWidth(clampPanelWidth(startWidth + startX - moveEvent.clientX));
    const stop = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", stop);
      window.document.body.classList.remove("resizing-panel");
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", stop);
    window.document.body.classList.add("resizing-panel");
  }

  function clampEditorHeight(height: number) {
    return Math.max(150, Math.min(height, Math.max(190, window.innerHeight - 320)));
  }

  function resizeEditorBy(delta: number) {
    setEditorHeight((height) => clampEditorHeight(height + delta));
  }

  function startEditorHeightResize(event: ReactPointerEvent<HTMLButtonElement>) {
    event.preventDefault();
    const startY = event.clientY;
    const startHeight = editorHeight;
    const move = (moveEvent: PointerEvent) => setEditorHeight(clampEditorHeight(startHeight + moveEvent.clientY - startY));
    const stop = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", stop);
      window.document.body.classList.remove("resizing-editor-height");
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", stop);
    window.document.body.classList.add("resizing-editor-height");
  }

  const activeBlockId = editing || formulaEditing?.blockId || null;
  const activeBlock = document?.blocks.find((block) => block.id === activeBlockId) || null;
  const reviewBodyStyle = activeBlock ? ({ "--review-panel-width": `${panelWidth}px` } as CSSProperties) : undefined;

  if (loading && !document) return <main className="center-state"><div className="spinner" /><p>Preparing the manuscript…</p></main>;
  if (error && !document) return <main className="center-state error-state"><h1>Review server is not running</h1><p>{error}</p><button onClick={load}>Try again</button></main>;

  return (
    <main className="workspace">
      <header className="topbar">
        <div className="brand"><p className="eyebrow">Hidden Policies · ICLR 2027</p><h1>Paper review</h1></div>
        <div className="review-summary">
          <span className="summary-chip"><strong>{Object.keys(proposals).length}</strong> pending</span>
          <span className="autosave-chip" title="Temporary edits are saved to a local file shared by browsers">Shared autosave</span>
          <span className={`status ${document?.clean ? "clean" : "dirty"}`}><span />{document?.clean ? "Manuscript clean" : "Uncommitted main.tex"}</span>
          <span className="commit-chip">{document?.branch} · {document?.head}</span>
          <button className="refresh-button" onClick={load} aria-label="Reload manuscript" title="Reload manuscript">↻</button>
        </div>
      </header>

      {(error || notice) && <div className={`toast ${error ? "toast-error" : "toast-success"}`} role="status">
        <span>{error || notice}</span><button onClick={() => { setError(""); setNotice(""); }} aria-label="Dismiss">×</button>
      </div>}

      <div className="editor-hint"><span>Click any paragraph or formula to edit in the side panel</span><span><kbd>⌘</kbd> + <kbd>Enter</kbd> to update · <kbd>Esc</kbd> to cancel</span></div>

      <div className={`review-body${activeBlock ? " panel-open" : ""}`} style={reviewBodyStyle}>
      <section className="paper-shell" aria-label="Tracked-change manuscript editor">
        {Array.from({ length: document?.pageCount || 1 }, (_, index) => index + 1).map((page) => (
          <article className="paper-page" key={page} aria-label={`Page ${page}`}>
            <div className="top-rule" />
            <div className="line-numbers" aria-hidden="true">{lineNumbers(page).map((line) => <span key={line}>{line}</span>)}</div>
            <div className="paper-content">
              {(pages.get(page) || []).map((block) => {
                const storedProposal = proposals[block.id];
                const isSelected = activeBlockId === block.id;
                let proposal = storedProposal;
                if (editing === block.id && draftPreview?.blockId === block.id)
                  proposal = draftPreview.updatedRaw === block.raw ? undefined : draftPreview;
                const continueEditing = () => {
                  if (isSelected) return;
                  if (block.kind === "display") startFormulaEdit(block, null); else startTextEdit(block);
                };
                return <div className={`${blockClass(block)}${proposal ? " has-change" : ""}${isSelected ? " editing-target" : ""}`} key={block.id} data-line={block.startLine}>
                  {proposal ? <>
                    <div className="diff-text" role="button" tabIndex={0} onClick={continueEditing}
                      onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); continueEditing(); } }}
                      title="Click to keep editing">{block.kind === "display" && proposal.mathSource
                        ? <><FormulaDiff before={mathSourceFromDisplay(block.raw)} after={proposal.mathSource} /><BlockPrefix block={block} /></>
                        : <><BlockPrefix block={block} /><DiffText before={block.text} after={proposal.newText} /></>}</div>
                    <div className="change-connector" aria-hidden="true" />
                    <div className="change-actions" aria-label="Review this change">
                      <button className="accept-icon" onClick={() => accept(block)} disabled={busy === block.id || isSelected} aria-label="Accept and commit" title={isSelected ? "Update or close the side editor first" : "Accept and commit"}>{busy === block.id ? "…" : "✓"}</button>
                      <button className="reject-icon" onClick={() => reject(block)} disabled={busy === block.id || isSelected} aria-label="Reject and restore" title={isSelected ? "Update or close the side editor first" : "Reject and restore"}>×</button>
                    </div>
                  </> : block.kind === "display" ? <div className="rendered-text rendered-formula" role="button" tabIndex={0}
                    onClick={() => { if (!isSelected) startFormulaEdit(block, null); }}
                    onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); startFormulaEdit(block, null); } }}>
                    <MathFormula source={mathSourceFromDisplay(block.raw)} displayMode /><BlockPrefix block={block} />
                  </div> : block.editable ? <div className="rendered-text" role="button" tabIndex={0}
                    onClick={() => { if (!isSelected) startTextEdit(block); }}
                    onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); startTextEdit(block); } }}>
                    <BlockText block={block} onEditMath={(index) => startFormulaEdit(block, index)} />
                  </div> : <div className="rendered-text"><BlockText block={block} onEditMath={(index) => startFormulaEdit(block, index)} /></div>}
                </div>;
              })}
            </div>
            <div className="page-number">{page}</div>
          </article>
        ))}
      </section>
      {activeBlock && <aside className="revision-panel" aria-label="Revision editor">
        <button type="button" className="panel-width-handle" aria-label="Resize revision panel" title="Drag to resize the revision panel"
          onPointerDown={startPanelResize}
          onKeyDown={(event) => {
            if (event.key === "ArrowLeft") { event.preventDefault(); setPanelWidth((width) => clampPanelWidth(width + 24)); }
            if (event.key === "ArrowRight") { event.preventDefault(); setPanelWidth((width) => clampPanelWidth(width - 24)); }
          }}><span /></button>
        <header className="revision-panel-header">
          <div><p>Revision editor</p><h2>{activeBlock.label} · line {activeBlock.startLine}</h2></div>
          <button type="button" onClick={() => { closeTextEditor(); setFormulaEditing(null); }} aria-label="Close editor" title="Close without updating">×</button>
        </header>
        <p className="revision-guidance">The tracked highlights stay visible on the paper while you edit here. Drag the left edge to change width and the grip below the editor up or down to change height.</p>
        {editing === activeBlock.id ? <>
          <div ref={(node) => {
            editorRef.current = node;
            if (node && node.dataset.editorFor !== activeBlock.id) {
              node.innerHTML = editorHtml(editingParts);
              node.dataset.editorFor = activeBlock.id;
            }
          }} className="inline-editor revision-input" style={{ height: editorHeight }} contentEditable role="textbox" aria-multiline="true" aria-label={`Edit ${activeBlock.label}`} tabIndex={0} suppressContentEditableWarning
            onInput={() => updateDraftPreview(activeBlock)}
            onKeyDown={(event) => {
              if (event.key === "Escape") { event.preventDefault(); closeTextEditor(); }
              if ((event.metaKey || event.ctrlKey) && event.key === "Enter") { event.preventDefault(); finishEditing(activeBlock); }
            }} />
          <EditorHeightHandle onPointerDown={startEditorHeightResize} onStep={resizeEditorBy} />
          <div className="edit-note">Protected citations, math, and emphasis stay intact.</div>
          <div className="revision-panel-footer">
            <button type="button" className="panel-cancel" onClick={closeTextEditor}>Cancel</button>
            <button type="button" className="panel-update" onClick={() => finishEditing(activeBlock)}>Update draft</button>
          </div>
        </> : formulaEditing?.blockId === activeBlock.id ? <FormulaEditor value={formulaEditing.value} displayMode={formulaEditing.partIndex === null} height={editorHeight}
          onChange={(value) => setFormulaEditing({ ...formulaEditing, value })}
          onDone={() => finishFormulaEdit(activeBlock)} onCancel={() => setFormulaEditing(null)}
          onResizeStart={startEditorHeightResize} onResizeStep={resizeEditorBy} /> : null}
      </aside>}
      </div>
    </main>
  );
}
