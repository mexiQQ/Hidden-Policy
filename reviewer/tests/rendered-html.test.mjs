import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);
  return worker.fetch(new Request("http://localhost/", { headers: { accept: "text/html" } }), {
    ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) },
  }, { waitUntil() {}, passThroughOnException() {} });
}

test("server-renders the paper reviewer shell", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);
  const html = await response.text();
  assert.match(html, /<title>Hidden Policies · Paper Review<\/title>/i);
  assert.match(html, /Preparing the manuscript/);
  assert.doesNotMatch(html, /codex-preview|react-loading-skeleton|Your site is taking shape/i);
});

test("includes tracked-change controls and the local commit bridge", async () => {
  const [editor, server, stylesheet] = await Promise.all([
    readFile(new URL("../app/PaperEditor.tsx", import.meta.url), "utf8"),
    readFile(new URL("../local-server.mjs", import.meta.url), "utf8"),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
  ]);
  assert.match(editor, /Accept and commit/);
  assert.match(editor, /className="panel-accept"/);
  assert.match(editor, /className="panel-reject"/);
  assert.match(editor, /Draft autosaved/);
  assert.match(editor, /sentenceUnits/);
  assert.match(editor, /className={`sentence-unit/);
  assert.doesNotMatch(editor, /Update draft/);
  assert.doesNotMatch(editor, /Click to edit this sentence/);
  assert.doesNotMatch(editor, /change-actions/);
  assert.match(stylesheet, /\.sentence-unit:hover/);
  assert.match(stylesheet, /\.sentence-unit\.sentence-selected/);
  assert.match(editor, /className="editor-height-handle"/);
  assert.match(stylesheet, /cursor:\s*ns-resize/);
  assert.match(editor, /aria-label="Apply bold"/);
  assert.match(editor, /execCommand\("bold"/);
  assert.match(editor, /\\\\textbf\{/);
  assert.match(editor, /aria-label="Apply italics"/);
  assert.match(editor, /execCommand\("italic"/);
  assert.match(editor, /\\\\emph\{/);
  assert.match(editor, /editable-italic/);
  assert.match(editor, /block\.kind === "quote"/);
  assert.match(editor, /replace\(\/\^\\\\begin/);
  assert.match(editor, /function editableRegion/);
  assert.match(editor, /function normalizeEditedRaw/);
  assert.match(editor, /block\.editable \|\| block\.kind === "quote"/);
  assert.match(editor, /aria-label="Copy sentence LaTeX"/);
  assert.match(editor, /aria-label="Copy paragraph LaTeX"/);
  assert.match(editor, /copyParagraphLatex/);
  assert.match(editor, /Copied the complete paragraph as LaTeX/);
  assert.match(editor, /navigator\.clipboard\?\.writeText/);
  assert.match(editor, /data-paper-block-id/);
  assert.match(editor, /ResizeObserver/);
  assert.match(editor, /keepWithNext/);
  assert.match(editor, /SENTENCE_LAYOUT_SEPARATOR/);
  assert.match(editor, /sentenceLayoutToken/);
  assert.match(editor, /data-paper-layout-key/);
  assert.match(editor, /fragment-continuation/);
  assert.match(editor, /followingMinimum/);
  assert.match(editor, /item\.block\.kind === "runin" \? 0/);
  assert.match(editor, /className="citation-link"/);
  assert.match(editor, /data-citation-token/);
  assert.match(editor, /citationsForChange/);
  assert.match(editor, /LinkedCitationText/);
  assert.match(editor, /\/open-external/);
  assert.match(server, /Google Chrome/);
  assert.match(server, /\["http:", "https:"\]/);
  assert.match(server, /spawnSync\("\/usr\/bin\/open", \["-a", "Google Chrome", url\.href\]/);
  assert.doesNotMatch(server, /calibratedPageStarts/);
  assert.match(server, /spawnSync\("make"/);
  assert.match(server, /const TEX_REPO_PATH = "paper\/main\.tex"/);
  assert.match(server, /\["commit", "-m", message, "--", TEX_REPO_PATH\]/);
  assert.match(server, /listen\(PORT, "127\.0\.0\.1"/);
});
