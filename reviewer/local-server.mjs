import { createHash } from "node:crypto";
import { createServer } from "node:http";
import { readFile, rename, unlink, writeFile } from "node:fs/promises";
import { spawnSync } from "node:child_process";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const REVIEWER_DIR = dirname(fileURLToPath(import.meta.url));
const REPO_DIR = resolve(REVIEWER_DIR, "..");
const TEX_PATH = resolve(REPO_DIR, "main.tex");
const BIB_PATH = resolve(REPO_DIR, "references.bib");
const LOG_PATH = resolve(REPO_DIR, "main.log");
const DRAFT_PATH = process.env.REVIEWER_DRAFT_PATH || resolve(REVIEWER_DIR, ".review-drafts.json");
const DRAFT_TEMP_PATH = `${DRAFT_PATH}.tmp`;
const PORT = Number(process.env.REVIEWER_API_PORT || 4318);
let draftSaveQueue = Promise.resolve();

function runGit(args) {
  return spawnSync("git", args, { cwd: REPO_DIR, encoding: "utf8" });
}

function commandOutput(args) {
  const result = runGit(args);
  if (result.status !== 0) throw new Error(result.stderr.trim() || `git ${args.join(" ")} failed`);
  return result.stdout.trim();
}

function parseBibliography(source) {
  const entries = new Map();
  const entryPattern = /@\w+\{([^,]+),([\s\S]*?)(?=\n@|\n?$)/g;
  for (const match of source.matchAll(entryPattern)) {
    const key = match[1].trim();
    const body = match[2];
    const author = body.match(/author\s*=\s*\{([^}]+)\}/i)?.[1] || key;
    const year = body.match(/year\s*=\s*\{?([^},\n]+)\}?/i)?.[1]?.trim() || "";
    const authors = author.split(/\s+and\s+/i);
    const first = authors[0].trim().split(/\s+/).at(-1)?.replace(/[{},]/g, "") || key;
    const label = `${first}${authors.length > 1 ? " et al." : ""}${year ? `, ${year}` : ""}`;
    const field = (name) => body.match(new RegExp(`^\\s*${name}\\s*=\\s*\\{(.*)\\}\\s*,?\\s*$`, "im"))?.[1]?.trim() || "";
    const title = field("title").replace(/[{}]/g, "");
    const url = field("url") || (field("doi") ? `https://doi.org/${field("doi")}` : "") ||
      (field("eprint") ? `https://arxiv.org/abs/${field("eprint")}` : "") ||
      `https://scholar.google.com/scholar?q=${encodeURIComponent(title || key)}`;
    entries.set(key, { label, title, url });
  }
  return entries;
}

function citationLabel(keys, bibliography, textual = false) {
  const labels = keys.split(",").map((key) => bibliography.get(key.trim())?.label || key.trim());
  if (!textual) return `(${labels.join("; ")})`;
  const [author, year] = (labels[0] || "").split(/,\s*/);
  return year ? `${author} (${year})` : author;
}

function citationTargets(keys, bibliography, textual = false) {
  const targets = keys.split(",").map((key) => {
    const trimmed = key.trim();
    const entry = bibliography.get(trimmed);
    return { display: entry?.label || trimmed, url: entry?.url || `https://scholar.google.com/scholar?q=${encodeURIComponent(trimmed)}` };
  });
  if (!textual) return targets;
  const first = targets[0];
  if (!first) return [];
  const [author, year] = first.display.split(/,\s*/);
  return [{ ...first, display: year ? `${author} (${year})` : author }];
}

function mathLabel(raw) {
  return raw
    .replace(/^\$|\$$/g, "")
    .replace(/\\pi/g, "π").replace(/\\tau/g, "τ").replace(/\\delta/g, "δ")
    .replace(/\\epsilon/g, "ε").replace(/\\approx/g, "≈").replace(/\\implies/g, "⇒")
    .replace(/\\centernot\s*/g, "⇏ ").replace(/\\mid/g, "|")
    .replace(/\\mathcal\{([^}]+)\}/g, "$1").replace(/\\text\{([^}]+)\}/g, "$1")
    .replace(/[{}]/g, "").replace(/\\,/g, " ").replace(/\\quad/g, " ")
    .replace(/\\([A-Za-z]+)/g, "$1");
}

function tokenizeInline(raw, bibliography, references = new Map()) {
  const parts = [];
  const pattern = /(\$[^$]+\$|\\(?:citep|citet|cite|ref|eqref)\{[^}]+\}|\\(?:emph|textit|textbf|textsc)\{[^}]+\}|\\(?:HP|Rint|Rreveal|Hist|Act|Traj)\b|~|---)/g;
  let cursor = 0;
  for (const match of raw.matchAll(pattern)) {
    if (match.index > cursor) parts.push({ type: "text", raw: raw.slice(cursor, match.index), display: raw.slice(cursor, match.index) });
    const token = match[0];
    let display = token;
    let style = "macro";
    let citations;
    let citationMode;
    let inner;
    if (token.startsWith("$")) { display = mathLabel(token); style = "math"; }
    else if ((inner = token.match(/^\\citep?\{([^}]+)\}$/))) {
      display = citationLabel(inner[1], bibliography); style = "citation";
      citations = citationTargets(inner[1], bibliography); citationMode = "parenthetical";
    }
    else if ((inner = token.match(/^\\citet\{([^}]+)\}$/))) {
      display = citationLabel(inner[1], bibliography, true); style = "citation";
      citations = citationTargets(inner[1], bibliography, true); citationMode = "textual";
    }
    else if ((inner = token.match(/^\\eqref\{([^}]+)\}$/))) { display = `(${references.get(inner[1]) || "?"})`; style = "reference"; }
    else if ((inner = token.match(/^\\ref\{([^}]+)\}$/))) { display = references.get(inner[1]) || inner[1]; style = "reference"; }
    else if ((inner = token.match(/^\\(?:emph|textit)\{([^}]+)\}$/))) { display = inner[1]; style = "italic"; }
    else if ((inner = token.match(/^\\textbf\{([^}]+)\}$/))) { display = inner[1]; style = "bold"; }
    else if ((inner = token.match(/^\\textsc\{([^}]+)\}$/))) { display = inner[1]; style = "smallcaps"; }
    else if (token === "~") { display = " "; style = "space"; }
    else if (token === "---") { display = "—"; style = "punctuation"; }
    else {
      const macros = { "\\HP": "Hidden Policy", "\\Rint": "R", "\\Rreveal": "R′", "\\Hist": "H", "\\Act": "A", "\\Traj": "T" };
      display = macros[token] || token;
    }
    parts.push({ type: "token", raw: token, display, style,
      ...(citations ? { citations, citationMode } : {}) });
    cursor = match.index + token.length;
  }
  if (cursor < raw.length) parts.push({ type: "text", raw: raw.slice(cursor), display: raw.slice(cursor) });
  return parts;
}

function plainText(raw, bibliography, references = new Map()) {
  return tokenizeInline(raw, bibliography, references).map((part) => part.display).join("")
    .replace(/\\[a-zA-Z*]+\{([^{}]*)\}/g, "$1")
    .replace(/\\[a-zA-Z*]+/g, "")
    .replace(/[{}]/g, "")
    .replace(/\s+/g, " ").trim();
}

function blockId(kind, startOffset, raw) {
  return createHash("sha256").update(`${kind}:${startOffset}:${raw}`).digest("hex").slice(0, 16);
}

function makeBlock({
  kind, raw, renderRaw = raw, startOffset, endOffset, startLine, bibliography, references,
  editable = true, label = kind, number, statementLabel, listType, items, segments,
}) {
  const id = blockId(kind, startOffset, raw);
  return {
    id, kind, raw, text: plainText(renderRaw, bibliography, references), parts: tokenizeInline(renderRaw, bibliography, references),
    startOffset, endOffset, startLine, endLine: startLine + raw.split("\n").length - 1,
    page: Math.max(1, Math.floor((startLine - 1) / 54) + 1), editable, label,
    ...(number ? { number } : {}), ...(statementLabel ? { statementLabel } : {}),
    ...(listType ? { listType } : {}), ...(items ? { items } : {}), ...(segments ? { segments } : {}),
  };
}

function displayMathSource(raw) {
  return raw
    .replace(/^\\begin\{(?:equation|align)\*?\}\s*/, "")
    .replace(/\s*\\end\{(?:equation|align)\*?\}\s*$/, "")
    .replace(/\\label\{[^}]+\}/g, "").trim();
}

function referenceLabels(source) {
  const references = new Map();
  let section = 0;
  let subsection = 0;
  let subsubsection = 0;
  let equation = 0;
  let theorem = 0;
  const lines = source.split("\n");
  for (let index = 0; index < lines.length; index++) {
    const trimmed = lines[index].trim();
    const heading = trimmed.match(/^\\(section|subsection|subsubsection)(\*)?\{/);
    let currentNumber = "";
    if (heading && !heading[2]) {
      if (heading[1] === "section") { section++; subsection = 0; subsubsection = 0; currentNumber = `${section}`; }
      if (heading[1] === "subsection") { subsection++; subsubsection = 0; currentNumber = `${section}.${subsection}`; }
      if (heading[1] === "subsubsection") { subsubsection++; currentNumber = `${section}.${subsection}.${subsubsection}`; }
    }
    const env = trimmed.match(/^\\begin\{(equation|align|definition|proposition)(\*)?\}/);
    if (env && !env[2]) {
      if (env[1] === "equation" || env[1] === "align") currentNumber = `${++equation}`;
      else currentNumber = `${++theorem}`;
    }
    if (!currentNumber) continue;
    if (env) {
      for (let cursor = index; cursor < lines.length; cursor++) {
        for (const match of lines[cursor].matchAll(/\\label\{([^}]+)\}/g)) references.set(match[1], currentNumber);
        if (lines[cursor].includes(`\\end{${env[1]}${env[2] || ""}}`)) break;
      }
      continue;
    }
    for (let cursor = index; cursor < Math.min(lines.length, index + 20); cursor++) {
      if (cursor > index && /^\\(?:section|subsection|subsubsection|begin)\b/.test(lines[cursor].trim())) break;
      for (const match of lines[cursor].matchAll(/\\label\{([^}]+)\}/g)) references.set(match[1], currentNumber);
      if (heading && cursor > index + 2) break;
    }
  }
  return references;
}

function findCommand(source, name) {
  const marker = `\\${name}{`;
  const start = source.indexOf(marker);
  if (start < 0) return null;
  let depth = 1;
  let cursor = start + marker.length;
  while (cursor < source.length && depth > 0) {
    if (source[cursor] === "{" && source[cursor - 1] !== "\\") depth++;
    if (source[cursor] === "}" && source[cursor - 1] !== "\\") depth--;
    cursor++;
  }
  return { raw: source.slice(start + marker.length, cursor - 1), startOffset: start + marker.length, endOffset: cursor - 1 };
}

function parseDocument(source, bibliography) {
  const blocks = [];
  const references = referenceLabels(source);
  let section = 0;
  let subsection = 0;
  let subsubsection = 0;
  let theorem = 0;
  const lineAt = (offset) => source.slice(0, offset).split("\n").length;
  const title = findCommand(source, "title");
  if (title) blocks.push(makeBlock({ kind: "title", ...title, startLine: 1, bibliography, references, label: "title" }));
  const author = findCommand(source, "author");
  if (author) blocks.push(makeBlock({ kind: "author", ...author, startLine: 5, bibliography, references, label: "author" }));

  const documentStart = source.indexOf("\\begin{document}");
  const bibliographyStart = source.indexOf("\\bibliographystyle", documentStart);
  const bodyStart = source.indexOf("\n", documentStart) + 1;
  const bodyEnd = bibliographyStart > 0 ? bibliographyStart : source.indexOf("\\end{document}", bodyStart);
  const body = source.slice(bodyStart, bodyEnd);
  const equationNumberAt = (offset) => String((source.slice(documentStart, offset).match(/\\begin\{(?:equation|align)\}/g) || []).length + 1);
  const lines = body.split("\n");
  const offsets = [];
  let running = bodyStart;
  for (const line of lines) { offsets.push(running); running += line.length + 1; }

  const specialStart = (line) => /^(\\(?:section|subsection|subsubsection|paragraph)\*?\{|\\begin\{)/.test(line.trim());
  const environmentEnd = (name) => `\\end{${name}}`;

  for (let i = 0; i < lines.length;) {
    const trimmed = lines[i].trim();
    if (!trimmed || trimmed.startsWith("%") || trimmed === "\\maketitle" || /^\\label\{/.test(trimmed)) { i++; continue; }
    const env = trimmed.match(/^\\begin\{([^}]+)\}/)?.[1];
    if (env) {
      const baseEnv = env.replace(/\*$/, "");
      let j = i;
      while (j < lines.length && !lines[j].includes(environmentEnd(env))) j++;
      const endIndex = Math.min(j, lines.length - 1);
      const fullRaw = lines.slice(i, endIndex + 1).join("\n");
      const startOffset = offsets[i];
      const endOffset = offsets[endIndex] + lines[endIndex].length;
      if (baseEnv === "abstract") {
        const prefix = `\\begin{abstract}\n`;
        const suffix = `\n\\end{abstract}`;
        const raw = fullRaw.slice(prefix.length, fullRaw.length - suffix.length);
        blocks.push(makeBlock({ kind: "abstract", raw, startOffset: startOffset + prefix.length, endOffset: endOffset - suffix.length, startLine: lineAt(startOffset), bibliography, references, label: "abstract" }));
      } else {
        const isMathEnvironment = baseEnv === "equation" || baseEnv === "align";
        if (isMathEnvironment) {
          const numbered = !env.endsWith("*");
          blocks.push(makeBlock({ kind: "display", raw: fullRaw, startOffset, endOffset, startLine: lineAt(startOffset), bibliography, references, editable: true, label: baseEnv, number: numbered ? equationNumberAt(startOffset) : undefined }));
        } else if (baseEnv === "quote") {
          const renderRaw = fullRaw.replace(/^\\begin\{quote\}\s*/, "").replace(/\s*\\end\{quote\}\s*$/, "").trim();
          blocks.push(makeBlock({ kind: "quote", raw: fullRaw, renderRaw, startOffset, endOffset, startLine: lineAt(startOffset), bibliography, references, editable: false, label: "quote" }));
        } else if (baseEnv === "definition" || baseEnv === "proposition") {
          const optionalTitle = trimmed.match(/^\\begin\{[^}]+\}(?:\[([^\]]+)\])?/)?.[1];
          const renderRaw = fullRaw.replace(/^\\begin\{[^}]+\}(?:\[[^\]]+\])?\s*/, "").replace(new RegExp(`\\s*\\\\end\\{${env}\\}\\s*$`), "").trim();
          const name = baseEnv === "definition" ? "Definition" : "Proposition";
          const statementLabel = `${name} ${++theorem}${optionalTitle ? ` (${optionalTitle})` : ""}.`;
          const segments = [];
          const mathPattern = /\\begin\{(equation|align)\*?\}[\s\S]*?\\end\{\1\*?\}/g;
          let segmentCursor = 0;
          for (const match of renderRaw.matchAll(mathPattern)) {
            const textRaw = renderRaw.slice(segmentCursor, match.index).trim();
            if (textRaw) segments.push({ type: "text", parts: tokenizeInline(textRaw, bibliography, references) });
            const mathOffset = startOffset + fullRaw.indexOf(match[0]);
            segments.push({ type: "math", source: displayMathSource(match[0]), number: match[0].startsWith(`\\begin{${match[1]}*}`) ? undefined : equationNumberAt(mathOffset) });
            segmentCursor = match.index + match[0].length;
          }
          const trailingRaw = renderRaw.slice(segmentCursor).trim();
          if (trailingRaw) segments.push({ type: "text", parts: tokenizeInline(trailingRaw, bibliography, references) });
          blocks.push(makeBlock({ kind: "statement", raw: fullRaw, renderRaw, startOffset, endOffset, startLine: lineAt(startOffset), bibliography, references, editable: false, label: env, statementLabel, segments }));
        } else if (baseEnv === "enumerate" || baseEnv === "itemize") {
          const inner = fullRaw.replace(/^\\begin\{[^}]+\}\s*/, "").replace(new RegExp(`\\s*\\\\end\\{${env}\\}\\s*$`), "").trim();
          const itemRaw = inner.split(/\s*\\item\s+/).filter(Boolean);
          const items = itemRaw.map((item) => ({ text: plainText(item, bibliography, references), parts: tokenizeInline(item, bibliography, references) }));
          blocks.push(makeBlock({ kind: "list", raw: fullRaw, renderRaw: inner, startOffset, endOffset, startLine: lineAt(startOffset), bibliography, references, editable: false, label: baseEnv, listType: baseEnv === "enumerate" ? "ordered" : "unordered", items }));
        } else {
          const renderRaw = fullRaw.replace(/^\\begin\{[^}]+\}\s*/, "").replace(new RegExp(`\\s*\\\\end\\{${env}\\}\\s*$`), "").trim();
          blocks.push(makeBlock({ kind: "statement", raw: fullRaw, renderRaw, startOffset, endOffset, startLine: lineAt(startOffset), bibliography, references, editable: false, label: env }));
        }
      }
      i = endIndex + 1;
      continue;
    }
    const heading = trimmed.match(/^\\(section|subsection|subsubsection|paragraph)(\*)?\{([\s\S]*)\}$/);
    if (heading) {
      const kind = heading[1] === "paragraph" ? "runin" : heading[1];
      let number;
      if (!heading[2] && heading[1] !== "paragraph") {
        if (heading[1] === "section") { section++; subsection = 0; subsubsection = 0; number = `${section}`; }
        if (heading[1] === "subsection") { subsection++; subsubsection = 0; number = `${section}.${subsection}`; }
        if (heading[1] === "subsubsection") { subsubsection++; number = `${section}.${subsection}.${subsubsection}`; }
      }
      const rawStartInLine = lines[i].indexOf("{") + 1;
      const raw = heading[3];
      blocks.push(makeBlock({ kind, raw, startOffset: offsets[i] + rawStartInLine, endOffset: offsets[i] + rawStartInLine + raw.length, startLine: lineAt(offsets[i]), bibliography, references, label: kind, number }));
      i++;
      continue;
    }
    if (trimmed.startsWith("\\[") || trimmed.startsWith("\\(")) {
      let j = i;
      const close = trimmed.startsWith("\\[") ? "\\]" : "\\)";
      while (j < lines.length && !lines[j].includes(close)) j++;
      const endIndex = Math.min(j, lines.length - 1);
      blocks.push(makeBlock({ kind: "display", raw: lines.slice(i, endIndex + 1).join("\n"), startOffset: offsets[i], endOffset: offsets[endIndex] + lines[endIndex].length, startLine: lineAt(offsets[i]), bibliography, references, editable: true, label: "display math" }));
      i = endIndex + 1;
      continue;
    }
    let j = i + 1;
    while (j < lines.length && lines[j].trim() && !specialStart(lines[j])) j++;
    const raw = lines.slice(i, j).join("\n");
    blocks.push(makeBlock({ kind: "paragraph", raw, startOffset: offsets[i], endOffset: offsets[j - 1] + lines[j - 1].length, startLine: lineAt(offsets[i]), bibliography, references, label: `paragraph near line ${lineAt(offsets[i])}` }));
    i = j;
  }
  return blocks;
}

async function documentPayload() {
  const [source, bibSource, logSource] = await Promise.all([
    readFile(TEX_PATH, "utf8"), readFile(BIB_PATH, "utf8"), readFile(LOG_PATH, "utf8").catch(() => ""),
  ]);
  const bibliography = parseBibliography(bibSource);
  const blocks = parseDocument(source, bibliography);
  const pageCount = Number(logSource.match(/Output written on main\.pdf \((\d+) pages?/)?.[1] || 7);
  // This is only an initial, pre-measurement estimate. The browser repaginates
  // blocks from their rendered heights so source-code line breaks cannot leave
  // a mostly empty paper card.
  const lastLine = Math.max(1, ...blocks.map((block) => block.endLine));
  const sourceLinesPerPage = Math.ceil(lastLine / pageCount);
  for (const block of blocks) {
    block.page = Math.min(pageCount, Math.max(1, Math.floor((block.startLine - 1) / sourceLinesPerPage) + 1));
  }
  const mainStatus = commandOutput(["status", "--porcelain", "--", "main.tex"]);
  return {
    title: "Hidden Policies: The Risk Behind Frontier Risks",
    branch: commandOutput(["branch", "--show-current"]),
    head: commandOutput(["rev-parse", "--short", "HEAD"]),
    clean: !mainStatus,
    blocks,
    pageCount,
  };
}

function json(response, status, data) {
  response.writeHead(status, { "content-type": "application/json; charset=utf-8", "cache-control": "no-store", "access-control-allow-origin": "http://localhost:3000", "vary": "origin" });
  response.end(JSON.stringify(data));
}

async function readJson(request) {
  let body = "";
  for await (const chunk of request) {
    body += chunk;
    if (body.length > 2_000_000) throw new Error("Request is too large");
  }
  return JSON.parse(body || "{}");
}

function allowedOrigin(request) {
  const origin = request.headers.origin;
  return !origin || origin === "http://localhost:3000" || origin === "http://127.0.0.1:3000";
}

function normalizeDraftPayload(payload) {
  if (!payload || payload.version !== 1 || !Array.isArray(payload.drafts)) throw new Error("Invalid draft payload");
  const drafts = payload.drafts.map((draft) => {
    if (!draft || typeof draft.blockId !== "string" || typeof draft.originalRaw !== "string" ||
      typeof draft.kind !== "string" || typeof draft.label !== "string" || !Number.isFinite(draft.startLine) ||
      !draft.proposal || typeof draft.proposal.updatedRaw !== "string" || typeof draft.proposal.newText !== "string") {
      throw new Error("Invalid saved draft");
    }
    return {
      blockId: draft.blockId,
      originalRaw: draft.originalRaw,
      kind: draft.kind,
      label: draft.label,
      startLine: draft.startLine,
      proposal: {
        updatedRaw: draft.proposal.updatedRaw,
        newText: draft.proposal.newText,
        ...(typeof draft.proposal.mathSource === "string" ? { mathSource: draft.proposal.mathSource } : {}),
      },
    };
  });
  return {
    version: 1,
    head: typeof payload.head === "string" ? payload.head : "",
    savedAt: new Date().toISOString(),
    drafts,
  };
}

async function readDraftPayload() {
  try {
    return normalizeDraftPayload(JSON.parse(await readFile(DRAFT_PATH, "utf8")));
  } catch (error) {
    if (error?.code !== "ENOENT") console.warn(`Ignoring unreadable draft file: ${error instanceof Error ? error.message : error}`);
    return { version: 1, head: "", savedAt: "", drafts: [] };
  }
}

function saveDraftPayload(payload) {
  const normalized = normalizeDraftPayload(payload);
  const operation = draftSaveQueue.then(async () => {
    if (!normalized.drafts.length) {
      await unlink(DRAFT_PATH).catch((error) => { if (error?.code !== "ENOENT") throw error; });
      await unlink(DRAFT_TEMP_PATH).catch((error) => { if (error?.code !== "ENOENT") throw error; });
      return normalized;
    }
    await writeFile(DRAFT_TEMP_PATH, `${JSON.stringify(normalized, null, 2)}\n`, { encoding: "utf8", mode: 0o600 });
    await rename(DRAFT_TEMP_PATH, DRAFT_PATH);
    return normalized;
  });
  draftSaveQueue = operation.catch(() => {});
  return operation;
}

async function acceptChange(payload) {
  if (!payload?.blockId || typeof payload.updatedRaw !== "string") throw new Error("Invalid change payload");
  if (commandOutput(["status", "--porcelain", "--", "main.tex"])) throw new Error("main.tex has uncommitted changes; commit or restore them before accepting a review change");
  const [source, bibSource] = await Promise.all([readFile(TEX_PATH, "utf8"), readFile(BIB_PATH, "utf8")]);
  const bibliography = parseBibliography(bibSource);
  const block = parseDocument(source, bibliography).find((item) => item.id === payload.blockId);
  if (!block) throw new Error("This paragraph changed since the review loaded. Refresh and try again.");
  const updatedSource = source.slice(0, block.startOffset) + payload.updatedRaw + source.slice(block.endOffset);
  if (updatedSource === source) throw new Error("No change to accept");
  await writeFile(TEX_PATH, updatedSource, "utf8");
  const build = spawnSync("make", [], { cwd: REPO_DIR, encoding: "utf8", timeout: 120_000 });
  if (build.status !== 0) {
    await writeFile(TEX_PATH, source, "utf8");
    throw new Error(`LaTeX validation failed; the original text was restored.\n${(build.stderr || build.stdout).slice(-1200)}`);
  }
  const add = runGit(["add", "--", "main.tex"]);
  if (add.status !== 0) { await writeFile(TEX_PATH, source, "utf8"); throw new Error(add.stderr.trim() || "Could not stage main.tex"); }
  const message = `Revise manuscript: ${block.label}`;
  const commit = runGit(["commit", "-m", message, "--", "main.tex"]);
  if (commit.status !== 0) {
    runGit(["restore", "--staged", "--", "main.tex"]);
    await writeFile(TEX_PATH, source, "utf8");
    throw new Error(commit.stderr.trim() || "Could not create commit");
  }
  return { commit: commandOutput(["rev-parse", "--short", "HEAD"]), message, document: await documentPayload() };
}

function openExternal(payload) {
  if (!payload || typeof payload.url !== "string") throw new Error("Invalid external link");
  let url;
  try { url = new URL(payload.url); } catch { throw new Error("Invalid external link"); }
  if (!new Set(["http:", "https:"]).has(url.protocol)) throw new Error("Only web links can be opened");
  const result = spawnSync("/usr/bin/open", ["-a", "Google Chrome", url.href], { encoding: "utf8", timeout: 10_000 });
  if (result.status !== 0) throw new Error(result.stderr.trim() || "Google Chrome could not be opened");
  return { ok: true };
}

const server = createServer(async (request, response) => {
  if (!allowedOrigin(request)) return json(response, 403, { error: "Origin not allowed" });
  if (request.method === "OPTIONS") {
    response.writeHead(204, { "access-control-allow-origin": request.headers.origin || "http://localhost:3000", "access-control-allow-methods": "GET,POST,OPTIONS", "access-control-allow-headers": "content-type", "vary": "origin" });
    return response.end();
  }
  try {
    if (request.method === "GET" && request.url === "/api/health") return json(response, 200, { ok: true });
    if (request.method === "GET" && request.url === "/api/document") return json(response, 200, await documentPayload());
    if (request.method === "GET" && request.url === "/api/drafts") return json(response, 200, await readDraftPayload());
    if (request.method === "POST" && request.url === "/api/drafts") return json(response, 200, await saveDraftPayload(await readJson(request)));
    if (request.method === "POST" && request.url === "/api/open-external") return json(response, 200, openExternal(await readJson(request)));
    if (request.method === "POST" && request.url === "/api/accept") return json(response, 200, await acceptChange(await readJson(request)));
    return json(response, 404, { error: "Not found" });
  } catch (error) {
    return json(response, 400, { error: error instanceof Error ? error.message : "Unexpected error" });
  }
});

server.listen(PORT, "127.0.0.1", () => {
  console.log(`Paper review API ready on http://127.0.0.1:${PORT}`);
});
