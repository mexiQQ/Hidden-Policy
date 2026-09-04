"use client";

import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type CSSProperties, type PointerEvent as ReactPointerEvent } from "react";
import katex from "katex";

type CitationTarget = { display: string; url: string };
type Part = {
  type: "text" | "token"; raw: string; display: string; style?: string;
  citations?: CitationTarget[]; citationMode?: "parenthetical" | "textual";
};
type Block = {
  id: string; kind: string; raw: string; text: string; parts: Part[]; startLine: number;
  endLine: number; page: number; editable: boolean; label: string; number?: string; statementLabel?: string;
  listType?: "ordered" | "unordered"; items?: Array<{ text: string; parts: Part[] }>;
  segments?: Array<{ type: "text"; parts: Part[] } | { type: "math"; source: string; number?: string }>;
};
type DocumentData = { title: string; branch: string; head: string; clean: boolean; pageCount: number; blocks: Block[] };
type Proposal = { updatedRaw: string; newText: string; mathSource?: string };
type DiffPart = { type: "same" | "insert" | "delete"; value: string };
type FormulaEditing = { blockId: string; partIndex: number | null; value: string; baseRaw: string };
type SentenceUnit = { raw: string; suffix: string; start: number; end: number; text: string; parts: Part[] };
type LayoutItem = { token: string; block: Block; sentenceStart?: number; sentenceEnd?: number };
type TextEditing = {
  blockId: string; sentenceIndex: number; editorKey: string; prefix: string; suffix: string; originalSentenceRaw: string;
};
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
    parts.push({ type: "token", raw: token, display, style,
      ...(known?.citations ? { citations: known.citations, citationMode: known.citationMode } : {}) });
    cursor = match.index + token.length;
  }
  if (cursor < raw.length) parts.push({ type: "text", raw: raw.slice(cursor), display: raw.slice(cursor) });
  return parts;
}

function textFromParts(parts: Part[]) {
  return parts.map((part) => part.display).join("").replace(/\s+/g, " ").trim();
}

function sentenceUnits(raw: string, fallback: Part[] = []): SentenceUnit[] {
  const units: SentenceUnit[] = [];
  const boundary = /[.!?](?:["')\]}]*)\s+(?=(?:[A-Z0-9]|\\[A-Z]|["'“‘]))/g;
  let start = 0;
  for (const match of raw.matchAll(boundary)) {
    const whitespace = match[0].match(/\s+$/)?.[0] || "";
    const end = (match.index || 0) + match[0].length - whitespace.length;
    const sentenceRaw = raw.slice(start, end);
    const parts = partsFromRaw(sentenceRaw, fallback);
    if (sentenceRaw) units.push({ raw: sentenceRaw, suffix: whitespace, start, end, text: textFromParts(parts), parts });
    start = end + whitespace.length;
  }
  const sentenceRaw = raw.slice(start);
  if (sentenceRaw || !units.length) {
    const parts = partsFromRaw(sentenceRaw, fallback);
    units.push({ raw: sentenceRaw, suffix: "", start, end: raw.length, text: textFromParts(parts), parts });
  }
  return units;
}

function replaceSentence(raw: string, unit: SentenceUnit, updatedRaw: string) {
  return raw.slice(0, unit.start) + updatedRaw + raw.slice(unit.end);
}

const SENTENCE_LAYOUT_SEPARATOR = "::sentences:";

function sentenceLayoutToken(blockId: string, start: number, end: number) {
  return `${blockId}${SENTENCE_LAYOUT_SEPARATOR}${start}:${end}`;
}

function parseLayoutToken(token: string) {
  const [blockId, range] = token.split(SENTENCE_LAYOUT_SEPARATOR);
  if (!range) return { blockId };
  const [sentenceStart, sentenceEnd] = range.split(":").map(Number);
  return Number.isFinite(sentenceStart) && Number.isFinite(sentenceEnd)
    ? { blockId, sentenceStart, sentenceEnd }
    : { blockId };
}

function editableRegion(block: Block, raw: string, fallbackRaw = "") {
  if (block.kind !== "quote") return { prefix: "", content: raw, suffix: "" };
  const splitQuote = (value: string) => {
    const opening = value.match(/^\\begin\{quote\}\s*/);
    const closing = value.match(/\s*\\end\{quote\}\s*$/);
    if (!opening || !closing || closing.index === undefined || closing.index < opening[0].length) return null;
    return {
      prefix: value.slice(0, opening[0].length),
      content: value.slice(opening[0].length, closing.index),
      suffix: value.slice(closing.index),
    };
  };
  const region = splitQuote(raw);
  if (region) return region;
  const fallback = splitQuote(fallbackRaw);
  return fallback ? { prefix: fallback.prefix, content: raw, suffix: fallback.suffix } : { prefix: "", content: raw, suffix: "" };
}

function normalizeEditedRaw(block: Block, raw: string) {
  if (block.kind === "quote" && !editableRegion(block, raw, block.raw).content.trim()) return "";
  return raw;
}

function textFromBlockRaw(block: Block, raw: string) {
  return textFromParts(partsFromRaw(editableRegion(block, raw, block.raw).content, block.parts));
}

function partIndexAtOffset(raw: string, fallback: Part[], offset: number) {
  const parts = partsFromRaw(raw, fallback);
  let cursor = 0;
  for (let index = 0; index < parts.length; index++) {
    const end = cursor + parts[index].raw.length;
    if (offset >= cursor && offset < end) return index;
    cursor = end;
  }
  return -1;
}

function editorHtml(parts: Part[]) {
  return parts.map((part, index) => part.type === "text"
    ? escapeHtml(part.display)
    : part.style === "citation" && part.citations?.length
      ? `<span contenteditable="false" class="locked-token token-citation" data-token="${index}" title="Click a reference to open it in Google Chrome">${part.citationMode === "parenthetical" ? "(" : ""}${part.citations.map((citation, citationIndex) => `${citationIndex ? "; " : ""}<button type="button" class="citation-link" data-citation-token="${index}" data-citation-index="${citationIndex}">${escapeHtml(citation.display)}</button>`).join("")}${part.citationMode === "parenthetical" ? ")" : ""}</span>`
    : part.style === "bold"
      ? `<strong class="editable-bold">${escapeHtml(part.display)}</strong>`
    : part.style === "italic"
      ? `<em class="editable-italic">${escapeHtml(part.display)}</em>`
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
    if (node.tagName === "STRONG" || node.tagName === "B" || node.style.fontWeight === "bold" || Number(node.style.fontWeight) >= 600)
      return `\\textbf{${inside}}`;
    if (node.tagName === "EM" || node.tagName === "I" || node.style.fontStyle === "italic")
      return `\\emph{${inside}}`;
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

function draftEntries(documentData: DocumentData, proposals: Record<string, Proposal>) {
  return Object.entries(proposals).map(([blockId, proposal]) => {
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

function makeDraftPayload(documentData: DocumentData, proposals: Record<string, Proposal>): StoredDraftPayload {
  return { version: 1, head: documentData.head, savedAt: new Date().toISOString(), drafts: draftEntries(documentData, proposals) };
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

function CitationLink({ citation, onOpenCitation }: { citation: CitationTarget; onOpenCitation: (target: CitationTarget) => void }) {
  return <button type="button" className="citation-link"
    onClick={(event) => { event.stopPropagation(); onOpenCitation(citation); }}
    aria-label={`Open ${citation.display} in Google Chrome`} title="Open reference in Google Chrome">{citation.display}</button>;
}

function PartsText({ parts, onEditMath, onOpenCitation }: {
  parts: Part[]; onEditMath?: (index: number) => void; onOpenCitation?: (target: CitationTarget) => void;
}) {
  return <>{parts.map((part, index) => part.type === "text"
    ? <span key={index}>{part.display}</span>
    : part.style === "math" && onEditMath ? <button key={index} type="button" className="inline-math-button"
      onClick={(event) => { event.stopPropagation(); onEditMath(index); }} aria-label={`Edit formula ${part.display}`} title="Click to edit formula">
      <MathFormula source={part.raw.slice(1, -1)} />
    </button>
      : part.style === "math" ? <span key={index} className="inline-math-static"><MathFormula source={part.raw.slice(1, -1)} /></span>
      : part.style === "citation" && part.citations?.length && onOpenCitation
        ? <span key={index} className="citation-group">{part.citationMode === "parenthetical" && "("}
          {part.citations.map((citation, citationIndex) => <span key={citation.url}>
            {citationIndex > 0 && "; "}<CitationLink citation={citation} onOpenCitation={onOpenCitation} />
          </span>)}{part.citationMode === "parenthetical" && ")"}</span>
      : <span key={index} className={`token token-${part.style || "macro"}`}>{part.display}</span>)}</>;
}

function BlockPrefix({ block }: { block: Block }) {
  if (block.statementLabel) return <span className="statement-label">{block.statementLabel} </span>;
  if (!block.number) return null;
  return <span className={block.kind === "display" ? "equation-number" : "heading-number"}>{block.kind === "display" ? `(${block.number})` : block.number}</span>;
}

function BlockText({ block, onEditMath, onOpenCitation }: {
  block: Block; onEditMath: (index: number) => void; onOpenCitation: (target: CitationTarget) => void;
}) {
  if (block.items?.length) {
    const List = block.listType === "ordered" ? "ol" : "ul";
    return <List>{block.items.map((item, index) => <li key={index}><PartsText parts={item.parts} onOpenCitation={onOpenCitation} /></li>)}</List>;
  }
  if (block.segments?.length) return <><BlockPrefix block={block} />{block.segments.map((segment, index) => segment.type === "text"
    ? <span className="statement-text" key={index}><PartsText parts={segment.parts} onOpenCitation={onOpenCitation} /></span>
    : <span className="statement-math" key={index}><MathFormula source={segment.source} displayMode />{segment.number && <span className="equation-number">({segment.number})</span>}</span>)}</>;
  return <><BlockPrefix block={block} /><PartsText parts={block.parts} onEditMath={block.editable ? onEditMath : undefined} onOpenCitation={onOpenCitation} /></>;
}

function LinkedCitationText({ value, citations, onOpenCitation }: {
  value: string; citations: CitationTarget[]; onOpenCitation: (target: CitationTarget) => void;
}) {
  const byDisplay = new Map(citations.map((citation) => [citation.display, citation]));
  const labels = Array.from(byDisplay.keys()).filter(Boolean).sort((left, right) => right.length - left.length);
  if (!labels.length) return <>{value}</>;
  const escaped = labels.map((label) => label.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const pieces = value.split(new RegExp(`(${escaped.join("|")})`, "g"));
  return <>{pieces.map((piece, index) => {
    const citation = byDisplay.get(piece);
    return citation ? <CitationLink key={`${piece}-${index}`} citation={citation} onOpenCitation={onOpenCitation} /> : <span key={index}>{piece}</span>;
  })}</>;
}

function DiffText({ before, after, citations, onOpenCitation }: {
  before: string; after: string; citations: CitationTarget[]; onOpenCitation: (target: CitationTarget) => void;
}) {
  return <>{wordDiff(before, after).map((part, index) => part.type === "same"
    ? <span key={index}><LinkedCitationText value={part.value} citations={citations} onOpenCitation={onOpenCitation} /></span>
    : part.type === "insert" ? <ins key={index}><LinkedCitationText value={part.value} citations={citations} onOpenCitation={onOpenCitation} /></ins>
      : <del key={index}><LinkedCitationText value={part.value} citations={citations} onOpenCitation={onOpenCitation} /></del>)}</>;
}

function citationsForChange(block: Block, proposal: Proposal) {
  const candidates = [...block.parts, ...partsFromRaw(proposal.updatedRaw, block.parts)].flatMap((part) => part.citations || []);
  return Array.from(new Map(candidates.map((citation) => [`${citation.display}\u0000${citation.url}`, citation])).values());
}

function FormattingDiff({ block, proposal, onOpenCitation }: { block: Block; proposal: Proposal; onOpenCitation: (target: CitationTarget) => void }) {
  const visibleRaw = block.kind === "quote"
    ? proposal.updatedRaw.replace(/^\\begin\{quote\}\s*/, "").replace(/\s*\\end\{quote\}\s*$/, "")
    : proposal.updatedRaw;
  const parts = partsFromRaw(visibleRaw, block.parts);
  return <>{parts.map((part, index) => {
    if (part.type === "text") return <span key={index}>{part.display}</span>;
    if (part.style === "bold") {
      const bold = <strong>{part.display}</strong>;
      return block.raw.includes(part.raw) ? <span key={index}>{bold}</span>
        : <ins key={index} className="format-change" title="Bold formatting added">{bold}</ins>;
    }
    if (part.style === "italic") {
      const italic = <em>{part.display}</em>;
      return block.raw.includes(part.raw) ? <span key={index}>{italic}</span>
        : <ins key={index} className="format-change format-change-italic" title="Italic formatting added">{italic}</ins>;
    }
    return <PartsText key={index} parts={[part]} onOpenCitation={onOpenCitation} />;
  })}</>;
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

function FormulaEditor({ value, displayMode, height, onChange, onClose, onResizeStart, onResizeStep }: {
  value: string; displayMode: boolean; height: number; onChange: (value: string) => void; onClose: () => void;
  onResizeStart: (event: ReactPointerEvent<HTMLButtonElement>) => void; onResizeStep: (delta: number) => void;
}) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  useEffect(() => { textareaRef.current?.focus(); }, []);
  return <div className={`formula-editor${displayMode ? " formula-editor-display" : ""}`}>
    <div className="formula-live-preview"><MathFormula source={value} displayMode={displayMode} /></div>
    <textarea ref={textareaRef} value={value} style={{ height }} onChange={(event) => onChange(event.target.value)}
      onKeyDown={(event) => {
        if (event.key === "Escape") { event.preventDefault(); onClose(); }
      }} aria-label="Edit LaTeX formula" spellCheck={false} />
    <EditorHeightHandle onPointerDown={onResizeStart} onStep={onResizeStep} />
    <div className="formula-editor-footer"><span>Edit LaTeX · live preview · autosaved</span></div>
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
  const [editing, setEditing] = useState<TextEditing | null>(null);
  const [formulaEditing, setFormulaEditing] = useState<FormulaEditing | null>(null);
  const [proposals, setProposals] = useState<Record<string, Proposal>>({});
  const [editingParts, setEditingParts] = useState<Part[]>([]);
  const [busy, setBusy] = useState<string | null>(null);
  const [notice, setNotice] = useState("");
  const [panelWidth, setPanelWidth] = useState(380);
  const [editorHeight, setEditorHeight] = useState(360);
  const [draftsReady, setDraftsReady] = useState(false);
  const [pageBlockIds, setPageBlockIds] = useState<string[][]>([]);
  const editorRef = useRef<HTMLDivElement>(null);
  const paperShellRef = useRef<HTMLElement>(null);

  const load = useCallback(async () => {
    setLoading(true); setError("");
    try {
      const response = await fetch(`${API}/document`, { cache: "no-store" });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Could not load the manuscript");
      setDocument(payload);
      setPageBlockIds([]);
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

  useEffect(() => {
    if (!draftsReady || !document) return;
    const payload = makeDraftPayload(document, proposals);
    const timer = window.setTimeout(() => {
      saveSharedDrafts(payload).catch((caught) => setError(caught instanceof Error ? caught.message : "Could not autosave review drafts"));
    }, 300);
    return () => window.clearTimeout(timer);
  }, [document, draftsReady, proposals]);

  const pages = useMemo<LayoutItem[][]>(() => {
    if (!document) return [];
    const byId = new Map(document.blocks.map((block) => [block.id, block]));
    const flattened = pageBlockIds.flat();
    const dynamicItems = flattened.map((token) => {
      const parsed = parseLayoutToken(token);
      const block = byId.get(parsed.blockId);
      return block ? { token, block, sentenceStart: parsed.sentenceStart, sentenceEnd: parsed.sentenceEnd } : null;
    }).filter((item): item is LayoutItem => Boolean(item));
    const representedIds = dynamicItems.map((item) => item.block.id)
      .filter((id, index, ids) => index === 0 || id !== ids[index - 1]);
    const hasCompleteDynamicLayout = dynamicItems.length === flattened.length &&
      representedIds.length === document.blocks.length && representedIds.every((id, index) => id === document.blocks[index].id);
    if (hasCompleteDynamicLayout) {
      let cursor = 0;
      return pageBlockIds.map((tokens) => tokens.map(() => dynamicItems[cursor++]));
    }

    const fallback = new Map<number, LayoutItem[]>();
    for (const block of document.blocks) {
      const page = Math.min(block.page, document.pageCount || 1);
      fallback.set(page, [...(fallback.get(page) || []), { token: block.id, block }]);
    }
    return Array.from(fallback.keys()).sort((left, right) => left - right).map((page) => fallback.get(page) || []);
  }, [document, pageBlockIds]);

  useLayoutEffect(() => {
    if (!document || !paperShellRef.current) return;
    const shell = paperShellRef.current;
    let frame = 0;

    const repaginate = () => {
      const elements = new Map<string, HTMLElement>();
      shell.querySelectorAll<HTMLElement>("[data-paper-layout-key]").forEach((element) => {
        const token = element.dataset.paperLayoutKey;
        if (token) elements.set(token, element);
      });
      const renderedItems = pages.flat();
      if (elements.size !== renderedItems.length) return;

      const firstContent = shell.querySelector<HTMLElement>(".paper-content");
      const capacity = Number.parseFloat(firstContent ? window.getComputedStyle(firstContent).minHeight : "") || 864;
      type Metric = {
        token: string; blockId: string; kind: string; height: number; marginTop: number; marginBottom: number;
        sentenceStart?: number; sentenceEnd?: number; lineHeight: number;
        sentences: Array<{ index: number; bottom: number }>;
      };
      const metrics: Metric[] = renderedItems.map((item) => {
        const element = elements.get(item.token)!;
        const style = window.getComputedStyle(element);
        const elementRect = element.getBoundingClientRect();
        const renderedText = element.querySelector<HTMLElement>(".rendered-text");
        const lineHeight = Number.parseFloat(renderedText ? window.getComputedStyle(renderedText).lineHeight : "") || 15;
        const sentences = Array.from(element.querySelectorAll<HTMLElement>(".sentence-unit")).map((sentence) => ({
          index: Number(sentence.dataset.sentenceIndex),
          bottom: sentence.getBoundingClientRect().bottom - elementRect.top,
        })).filter((sentence) => Number.isFinite(sentence.index));
        return {
          token: item.token,
          blockId: item.block.id,
          kind: item.block.kind,
          // A run-in heading and the paragraph after it share the same first line.
          // The paragraph's inline box already accounts for that line height.
          height: item.block.kind === "runin" ? 0 : Math.ceil(element.offsetHeight),
          marginTop: Number.parseFloat(style.marginTop) || 0,
          marginBottom: Number.parseFloat(style.marginBottom) || 0,
          sentenceStart: item.sentenceStart,
          sentenceEnd: item.sentenceEnd,
          lineHeight,
          sentences,
        };
      });

      const nextPages: string[][] = [];
      let current: string[] = [];
      let used = 0;
      let previousMarginBottom = 0;
      const keepWithNext = new Set(["section", "subsection", "subsubsection", "runin"]);

      const startNewPage = () => {
        if (current.length) nextPages.push(current);
        current = [];
        used = 0;
        previousMarginBottom = 0;
      };

      const queue = [...metrics];
      while (queue.length) {
        const metric = queue.shift()!;
        const gap = current.length ? Math.max(previousMarginBottom, metric.marginTop) : metric.marginTop;
        const projectedBottom = used + gap + metric.height;
        const following = queue[0];
        const followingMinimum = following?.sentences.length
          ? Math.min(following.height, Math.max(following.lineHeight * 2, following.sentences[0].bottom))
          : following?.height || 0;
        const keepTogetherHeight = keepWithNext.has(metric.kind) && following
          ? Math.max(metric.marginBottom, following.marginTop) + followingMinimum
          : metric.marginBottom;

        if (current.length && projectedBottom + keepTogetherHeight > capacity) {
          const available = capacity - used - gap;
          const fitting = metric.sentences.filter((sentence) => sentence.bottom <= available + 0.5);
          if (fitting.length && fitting.length < metric.sentences.length) {
            const lastFitting = fitting[fitting.length - 1];
            const end = metric.sentenceEnd ?? (metric.sentences.at(-1)!.index + 1);
            current.push(sentenceLayoutToken(metric.blockId, metric.sentenceStart ?? metric.sentences[0].index, lastFitting.index + 1));
            used += gap + lastFitting.bottom;
            previousMarginBottom = 0;
            startNewPage();
            const remainderSentences = metric.sentences.slice(fitting.length).map((sentence) => ({
              index: sentence.index,
              bottom: Math.max(metric.lineHeight, sentence.bottom - lastFitting.bottom + metric.lineHeight),
            }));
            queue.unshift({
              ...metric,
              token: sentenceLayoutToken(metric.blockId, lastFitting.index + 1, end),
              sentenceStart: lastFitting.index + 1,
              sentenceEnd: end,
              height: Math.max(metric.lineHeight, metric.height - lastFitting.bottom + metric.lineHeight),
              marginTop: 0,
              sentences: remainderSentences,
            });
            continue;
          }
          startNewPage();
          queue.unshift(metric);
          continue;
        }

        if (!current.length && projectedBottom + keepTogetherHeight > capacity && metric.sentences.length > 1) {
          const fitting = metric.sentences.filter((sentence) => sentence.bottom <= capacity + 0.5);
          if (fitting.length && fitting.length < metric.sentences.length) {
            const lastFitting = fitting[fitting.length - 1];
            const end = metric.sentenceEnd ?? (metric.sentences.at(-1)!.index + 1);
            current.push(sentenceLayoutToken(metric.blockId, metric.sentenceStart ?? metric.sentences[0].index, lastFitting.index + 1));
            startNewPage();
            queue.unshift({
              ...metric,
              token: sentenceLayoutToken(metric.blockId, lastFitting.index + 1, end),
              sentenceStart: lastFitting.index + 1,
              sentenceEnd: end,
              height: Math.max(metric.lineHeight, metric.height - lastFitting.bottom + metric.lineHeight),
              marginTop: 0,
              sentences: metric.sentences.slice(fitting.length).map((sentence) => ({
                index: sentence.index,
                bottom: Math.max(metric.lineHeight, sentence.bottom - lastFitting.bottom + metric.lineHeight),
              })),
            });
            continue;
          }
        }

        current.push(metric.token);
        used = projectedBottom;
        previousMarginBottom = metric.marginBottom;
      }
      startNewPage();

      setPageBlockIds((currentPages) => {
        const before = currentPages.map((page) => page.join("\u0000")).join("\u0001");
        const after = nextPages.map((page) => page.join("\u0000")).join("\u0001");
        return before === after ? currentPages : nextPages;
      });
    };

    const schedule = () => {
      window.cancelAnimationFrame(frame);
      frame = window.requestAnimationFrame(repaginate);
    };
    schedule();

    const observer = new ResizeObserver(schedule);
    shell.querySelectorAll<HTMLElement>("[data-paper-block-id]").forEach((element) => observer.observe(element));
    window.addEventListener("resize", schedule);
    return () => {
      window.cancelAnimationFrame(frame);
      observer.disconnect();
      window.removeEventListener("resize", schedule);
    };
  }, [document, pageBlockIds, pages, proposals]);

  function startTextEdit(block: Block, sentenceIndex: number) {
    const effectiveRaw = proposals[block.id]?.updatedRaw ?? block.raw;
    const effectiveRegion = editableRegion(block, effectiveRaw, block.raw);
    const units = sentenceUnits(effectiveRegion.content, block.parts);
    const unit = units[sentenceIndex];
    if (!unit) return;
    const originalRegion = editableRegion(block, block.raw);
    const originalUnit = sentenceUnits(originalRegion.content, block.parts)[sentenceIndex] || unit;
    setFormulaEditing(null);
    setEditingParts(unit.parts);
    setEditing({
      blockId: block.id, sentenceIndex, editorKey: `${block.id}:${sentenceIndex}`,
      prefix: effectiveRegion.prefix + effectiveRegion.content.slice(0, unit.start),
      suffix: effectiveRegion.content.slice(unit.end) + effectiveRegion.suffix,
      originalSentenceRaw: originalUnit.raw,
    });
  }

  function updateTextDraft(block: Block) {
    if (!editorRef.current || editing?.blockId !== block.id) return;
    const sentenceRaw = serializeEditor(editorRef.current, editingParts);
    const updatedRaw = normalizeEditedRaw(block, editing.prefix + sentenceRaw + editing.suffix);
    const newText = textFromBlockRaw(block, updatedRaw);
    setPageBlockIds([]);
    setProposals((current) => {
      if (updatedRaw === block.raw) {
        const next = { ...current }; delete next[block.id]; return next;
      }
      return { ...current, [block.id]: { updatedRaw, newText } };
    });
  }

  function toggleBold(block: Block) {
    const root = editorRef.current;
    const selection = window.getSelection();
    if (!root || !selection || !selection.rangeCount || selection.isCollapsed || !root.contains(selection.getRangeAt(0).commonAncestorContainer)) {
      setNotice("Select text in the revision editor before applying bold.");
      return;
    }
    const includesProtectedToken = Array.from(root.querySelectorAll(".locked-token"))
      .some((token) => selection.containsNode(token, true));
    if (includesProtectedToken) {
      setNotice("Bold can be applied to plain text, but not across protected citations, formulas, or references.");
      return;
    }
    window.document.execCommand("styleWithCSS", false, "false");
    if (!window.document.execCommand("bold", false)) {
      setNotice("Could not apply bold to this selection.");
      return;
    }
    root.focus();
    updateTextDraft(block);
  }

  function toggleItalic(block: Block) {
    const root = editorRef.current;
    const selection = window.getSelection();
    if (!root || !selection || !selection.rangeCount || selection.isCollapsed || !root.contains(selection.getRangeAt(0).commonAncestorContainer)) {
      setNotice("Select text in the revision editor before applying italics.");
      return;
    }
    const includesProtectedToken = Array.from(root.querySelectorAll(".locked-token"))
      .some((token) => selection.containsNode(token, true));
    if (includesProtectedToken) {
      setNotice("Italics can be applied to plain text, but not across protected citations, formulas, or references.");
      return;
    }
    window.document.execCommand("styleWithCSS", false, "false");
    if (!window.document.execCommand("italic", false)) {
      setNotice("Could not apply italics to this selection.");
      return;
    }
    root.focus();
    updateTextDraft(block);
  }

  async function copyLatex(value: string, successMessage: string) {
    setError("");
    try {
      let copied = false;
      if (navigator.clipboard?.writeText) {
        try {
          await navigator.clipboard.writeText(value);
          copied = true;
        } catch {
          // Fall back to the selection-based copy command below.
        }
      }
      if (!copied) {
        const textarea = window.document.createElement("textarea");
        textarea.value = value;
        textarea.setAttribute("readonly", "");
        textarea.style.position = "fixed";
        textarea.style.opacity = "0";
        window.document.body.appendChild(textarea);
        textarea.select();
        copied = window.document.execCommand("copy");
        textarea.remove();
      }
      if (!copied) throw new Error("Copy command was not available");
      setNotice(successMessage);
    } catch {
      setError("Could not copy LaTeX. Select the editor contents and copy them manually.");
    }
  }

  function copyTextEditorLatex() {
    if (!editorRef.current) return;
    void copyLatex(serializeEditor(editorRef.current, editingParts), "Copied the current sentence as LaTeX.");
  }

  function copyParagraphLatex(block: Block) {
    let paragraphRaw = proposals[block.id]?.updatedRaw ?? block.raw;
    if (editorRef.current && editing?.blockId === block.id) {
      paragraphRaw = editing.prefix + serializeEditor(editorRef.current, editingParts) + editing.suffix;
    }
    void copyLatex(paragraphRaw, "Copied the complete paragraph as LaTeX.");
  }

  async function openCitationInChrome(target: CitationTarget) {
    setError("");
    try {
      const response = await fetch(`${API}/open-external`, {
        method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ url: target.url }),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || "Could not open the reference");
      setNotice(`Opened ${target.display} in Google Chrome.`);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not open the reference in Google Chrome");
    }
  }

  function closeTextEditor() {
    setEditing(null);
    setEditingParts([]);
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

  function updateFormulaDraft(block: Block, value: string) {
    if (!formulaEditing || formulaEditing.blockId !== block.id) return;
    const nextEditing = { ...formulaEditing, value };
    setFormulaEditing(nextEditing);
    const proposal = proposalFromFormulaEdit(block, nextEditing);
    setPageBlockIds([]);
    if (proposal.updatedRaw === block.raw) setProposals((current) => { const next = { ...current }; delete next[block.id]; return next; });
    else setProposals((current) => ({ ...current, [block.id]: proposal }));
  }

  async function accept(block: Block) {
    const proposal = proposals[block.id];
    if (!proposal) return;
    let acceptedRaw = proposal.updatedRaw;
    let fullDraftRaw = proposal.updatedRaw;
    if (editing?.blockId === block.id) {
      const originalRegion = editableRegion(block, block.raw);
      const originalUnit = sentenceUnits(originalRegion.content, block.parts)[editing.sentenceIndex];
      const sentenceRaw = editorRef.current ? serializeEditor(editorRef.current, editingParts) : editing.originalSentenceRaw;
      fullDraftRaw = normalizeEditedRaw(block, editing.prefix + sentenceRaw + editing.suffix);
      if (originalUnit) acceptedRaw = normalizeEditedRaw(block,
        originalRegion.prefix + replaceSentence(originalRegion.content, originalUnit, sentenceRaw) + originalRegion.suffix);
    }
    setBusy(block.id); setError(""); setNotice("");
    try {
      const response = await fetch(`${API}/accept`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ blockId: block.id, updatedRaw: acceptedRaw }) });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Could not accept the change");
      setDocument(payload.document);
      setPageBlockIds([]);
      setProposals((current) => {
        if (!document) return {};
        const remaining = draftEntries(document, current).filter((entry) => entry.blockId !== block.id);
        const restored = proposalsFromEntries(payload.document, remaining);
        const updatedBlock = payload.document.blocks.find((candidate: Block) => candidate.raw === acceptedRaw) ||
          payload.document.blocks.find((candidate: Block) => candidate.kind === block.kind && candidate.label === block.label && Math.abs(candidate.startLine - block.startLine) <= 2);
        if (updatedBlock && fullDraftRaw !== acceptedRaw) {
          restored[updatedBlock.id] = { updatedRaw: fullDraftRaw, newText: textFromBlockRaw(updatedBlock, fullDraftRaw) };
        }
        return restored;
      });
      if (editing?.blockId === block.id) closeTextEditor();
      if (formulaEditing?.blockId === block.id) setFormulaEditing(null);
      setNotice(`Accepted this ${editing?.blockId === block.id ? "sentence" : "change"} and committed as ${payload.commit}`);
    } catch (caught) { setError(caught instanceof Error ? caught.message : "Could not accept the change"); }
    finally { setBusy(null); }
  }

  function reject(block: Block) {
    setPageBlockIds([]);
    setProposals((current) => {
      const next = { ...current };
      if (editing?.blockId === block.id) {
        const revertedRaw = normalizeEditedRaw(block, editing.prefix + editing.originalSentenceRaw + editing.suffix);
        if (revertedRaw === block.raw) delete next[block.id];
        else next[block.id] = { updatedRaw: revertedRaw, newText: textFromBlockRaw(block, revertedRaw) };
      } else delete next[block.id];
      return next;
    });
    closeTextEditor(); setFormulaEditing(null); setNotice(`Rejected this ${editing?.blockId === block.id ? "sentence" : "change"}; the original was restored.`);
  }

  function blockClass(block: Block) { return `paper-block kind-${block.kind}${block.editable || block.kind === "quote" ? " can-edit" : ""}`; }

  function sentenceEditable(block: Block) {
    return (block.editable || block.kind === "quote") && block.kind !== "display" && !block.items?.length && !block.segments?.length;
  }

  function sentenceContent(block: Block, proposal?: Proposal, sentenceStart = 0, sentenceEnd?: number) {
    const effectiveRaw = proposal?.updatedRaw ?? block.raw;
    const effectiveRegion = editableRegion(block, effectiveRaw, block.raw);
    const originalRegion = editableRegion(block, block.raw);
    const allUnits = sentenceUnits(effectiveRegion.content, block.parts);
    const originalUnits = sentenceUnits(originalRegion.content, block.parts);
    const boundedEnd = Math.min(sentenceEnd ?? allUnits.length, allUnits.length);
    const units = allUnits.slice(sentenceStart, boundedEnd);
    return <div className="rendered-text sentence-flow">{sentenceStart === 0 && <BlockPrefix block={block} />}{units.map((unit, localIndex) => {
      const sentenceIndex = sentenceStart + localIndex;
      const original = originalUnits[sentenceIndex] || unit;
      const changed = unit.raw !== original.raw;
      const selected = editing?.blockId === block.id && editing.sentenceIndex === sentenceIndex;
      const sentenceProposal = { updatedRaw: unit.raw, newText: unit.text };
      const sentenceBlock = { ...block, raw: original.raw, text: original.text, parts: original.parts };
      const edit = () => { if (!selected) startTextEdit(block, sentenceIndex); };
      return <span key={`${block.id}:${sentenceIndex}`}>
        <span className={`sentence-unit${changed ? " sentence-changed" : ""}${selected ? " sentence-selected" : ""}`}
          role="button" tabIndex={0} data-sentence-index={sentenceIndex}
          onClick={edit} onKeyDown={(event) => {
            if (event.key === "Enter" || event.key === " ") { event.preventDefault(); edit(); }
          }}>
          {changed ? (unit.text === original.text && unit.raw !== original.raw
            ? <FormattingDiff block={sentenceBlock} proposal={sentenceProposal} onOpenCitation={openCitationInChrome} />
            : <DiffText before={original.text} after={unit.text} citations={citationsForChange(sentenceBlock, sentenceProposal)} onOpenCitation={openCitationInChrome} />)
            : <PartsText parts={unit.parts} onOpenCitation={openCitationInChrome} onEditMath={(partIndex) => {
              const localOffset = unit.parts.slice(0, partIndex).reduce((total, part) => total + part.raw.length, 0);
              const blockPartIndex = partIndexAtOffset(effectiveRegion.content, block.parts, unit.start + localOffset);
              if (blockPartIndex >= 0) startFormulaEdit(block, blockPartIndex);
            }} />}
        </span>{unit.suffix}
      </span>;
    })}</div>;
  }

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

  const activeBlockId = editing?.blockId || formulaEditing?.blockId || null;
  const activeBlock = document?.blocks.find((block) => block.id === activeBlockId) || null;
  const activeProposal = activeBlock ? proposals[activeBlock.id] : undefined;
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

      <div className="editor-hint"><span>Hover or click a sentence to highlight and edit it</span><span>Changes save automatically · <kbd>Esc</kbd> closes the editor</span></div>

      <div className={`review-body${activeBlock ? " panel-open" : ""}`} style={reviewBodyStyle}>
      <section ref={paperShellRef} className="paper-shell" aria-label="Tracked-change manuscript editor">
        {pages.map((pageBlocks, index) => {
          const page = index + 1;
          return (
          <article className="paper-page" key={page} aria-label={`Page ${page}`}>
            <div className="top-rule" />
            <div className="line-numbers" aria-hidden="true">{lineNumbers(page).map((line) => <span key={line}>{line}</span>)}</div>
            <div className="paper-content">
              {pageBlocks.map((item) => {
                const { block, token, sentenceStart, sentenceEnd } = item;
                const proposal = proposals[block.id];
                const isSelected = activeBlockId === block.id;
                const effectiveRaw = proposal?.updatedRaw ?? block.raw;
                const sentenceCount = sentenceEditable(block)
                  ? sentenceUnits(editableRegion(block, effectiveRaw, block.raw).content, block.parts).length : 0;
                const continuesFromPreviousPage = sentenceStart !== undefined && sentenceStart > 0;
                const continuesOnNextPage = sentenceEnd !== undefined && sentenceEnd < sentenceCount;
                return <div className={`${blockClass(block)}${proposal ? " has-change" : ""}${isSelected ? " editing-target" : ""}${continuesFromPreviousPage ? " fragment-continuation" : ""}${continuesOnNextPage ? " fragment-continues" : ""}`} key={token}
                  data-line={block.startLine} data-paper-block-id={block.id} data-paper-layout-key={token}>
                  {sentenceEditable(block) ? sentenceContent(block, proposal, sentenceStart, sentenceEnd)
                    : proposal ?
                    <div className="diff-text" role="button" tabIndex={0}
                      onClick={() => { if (!isSelected && block.kind === "display") startFormulaEdit(block, null); }}
                      onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); if (block.kind === "display") startFormulaEdit(block, null); } }}
                      title="Click to keep editing">{block.kind === "display" && proposal.mathSource
                        ? <><FormulaDiff before={mathSourceFromDisplay(block.raw)} after={proposal.mathSource} /><BlockPrefix block={block} /></>
                        : <><BlockPrefix block={block} />{proposal.newText === block.text && proposal.updatedRaw !== block.raw
                          ? <FormattingDiff block={block} proposal={proposal} onOpenCitation={openCitationInChrome} />
                          : <DiffText before={block.text} after={proposal.newText} citations={citationsForChange(block, proposal)} onOpenCitation={openCitationInChrome} />}</>}</div>
                    : block.kind === "display" ? <div className="rendered-text rendered-formula" role="button" tabIndex={0}
                    onClick={() => { if (!isSelected) startFormulaEdit(block, null); }}
                    onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); startFormulaEdit(block, null); } }}>
                    <MathFormula source={mathSourceFromDisplay(block.raw)} displayMode /><BlockPrefix block={block} />
                  </div> : <div className="rendered-text"><BlockText block={block} onEditMath={(index) => startFormulaEdit(block, index)} onOpenCitation={openCitationInChrome} /></div>}
                </div>;
              })}
            </div>
            <div className="page-number">{page}</div>
          </article>
          );
        })}
      </section>
      {activeBlock && <aside className="revision-panel" aria-label="Revision editor">
        <button type="button" className="panel-width-handle" aria-label="Resize revision panel" title="Drag to resize the revision panel"
          onPointerDown={startPanelResize}
          onKeyDown={(event) => {
            if (event.key === "ArrowLeft") { event.preventDefault(); setPanelWidth((width) => clampPanelWidth(width + 24)); }
            if (event.key === "ArrowRight") { event.preventDefault(); setPanelWidth((width) => clampPanelWidth(width - 24)); }
          }}><span /></button>
        <header className="revision-panel-header">
          <div><p>Revision editor</p><h2>{editing?.blockId === activeBlock.id ? `Sentence ${editing.sentenceIndex + 1}` : activeBlock.label} · line {activeBlock.startLine}</h2></div>
          <button type="button" onClick={() => { closeTextEditor(); setFormulaEditing(null); }} aria-label="Close editor" title="Close editor">×</button>
        </header>
        <p className="revision-guidance">Edit this sentence while the tracked highlights remain visible on the paper. Changes are saved automatically.</p>
        {editing?.blockId === activeBlock.id ? <>
          <div className="editor-format-toolbar" role="toolbar" aria-label="Text formatting">
            <button type="button" className="format-bold" onPointerDown={(event) => event.preventDefault()}
              onClick={() => toggleBold(activeBlock)} aria-label="Apply bold" title="Bold selected text (⌘/Ctrl+B)"><strong>B</strong></button>
            <button type="button" className="format-italic" onPointerDown={(event) => event.preventDefault()}
              onClick={() => toggleItalic(activeBlock)} aria-label="Apply italics" title="Italicize selected text (⌘/Ctrl+I)"><em>I</em></button>
            <button type="button" className="copy-latex" onPointerDown={(event) => event.preventDefault()}
              onClick={copyTextEditorLatex} aria-label="Copy sentence LaTeX" title="Copy the current sentence as LaTeX">Copy sentence</button>
            <button type="button" className="copy-latex copy-paragraph" onPointerDown={(event) => event.preventDefault()}
              onClick={() => copyParagraphLatex(activeBlock)} aria-label="Copy paragraph LaTeX" title="Copy the complete paragraph as LaTeX">Copy paragraph</button>
            <span>Live edits included · drag the grip below to resize</span>
          </div>
          <div ref={(node) => {
            editorRef.current = node;
            if (node && node.dataset.editorFor !== editing.editorKey) {
              node.innerHTML = editorHtml(editingParts);
              node.dataset.editorFor = editing.editorKey;
            }
          }} className="inline-editor revision-input" style={{ height: editorHeight }} contentEditable role="textbox" aria-multiline="true" aria-label={`Edit sentence ${editing.sentenceIndex + 1} in ${activeBlock.label}`} tabIndex={0} suppressContentEditableWarning
            onClick={(event) => {
              const citationButton = (event.target as HTMLElement).closest<HTMLElement>("[data-citation-token]");
              if (!citationButton) return;
              event.preventDefault(); event.stopPropagation();
              const tokenIndex = Number(citationButton.dataset.citationToken);
              const citationIndex = Number(citationButton.dataset.citationIndex);
              const citation = editingParts[tokenIndex]?.citations?.[citationIndex];
              if (citation) void openCitationInChrome(citation);
            }}
            onInput={() => updateTextDraft(activeBlock)}
            onKeyDown={(event) => {
              if (event.key === "Escape") { event.preventDefault(); closeTextEditor(); }
              if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "b") { event.preventDefault(); toggleBold(activeBlock); }
              if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "i") { event.preventDefault(); toggleItalic(activeBlock); }
            }} />
          <EditorHeightHandle onPointerDown={startEditorHeightResize} onStep={resizeEditorBy} />
          <div className="edit-note">Bold is saved as LaTeX textbf; italics are saved as emph. Citations, formulas, and references remain protected.</div>
        </> : formulaEditing?.blockId === activeBlock.id ? <>
          <div className="editor-format-toolbar" role="toolbar" aria-label="Formula tools">
            <button type="button" className="copy-latex" onClick={() => void copyLatex(formulaEditing.value, "Copied the current formula as LaTeX.")}
              aria-label="Copy formula LaTeX" title="Copy the current formula source">Copy LaTeX</button>
            <span>Copies the current formula source</span>
          </div>
          <FormulaEditor value={formulaEditing.value} displayMode={formulaEditing.partIndex === null} height={editorHeight}
            onChange={(value) => updateFormulaDraft(activeBlock, value)} onClose={() => setFormulaEditing(null)}
            onResizeStart={startEditorHeightResize} onResizeStep={resizeEditorBy} />
        </> : null}
        <div className="revision-autosave-status"><span />{activeProposal ? "Draft autosaved" : "No changes yet"}</div>
        <div className="revision-panel-footer">
          <button type="button" className="panel-reject" onClick={() => reject(activeBlock)} disabled={!activeProposal || busy === activeBlock.id}
            aria-label="Reject this change" title="Reject and restore the original">× <span>Reject</span></button>
          <button type="button" className="panel-accept" onClick={() => accept(activeBlock)} disabled={!activeProposal || busy === activeBlock.id}
            aria-label="Accept and commit this change" title="Accept this change and create a commit">{busy === activeBlock.id ? "…" : "✓"} <span>Accept & commit</span></button>
        </div>
      </aside>}
      </div>
    </main>
  );
}
