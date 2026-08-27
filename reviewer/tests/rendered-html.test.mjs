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
  assert.match(editor, /Reject and restore/);
  assert.match(editor, /className="accept-icon"[\s\S]*?"✓"/);
  assert.match(editor, /className="reject-icon"[\s\S]*?>×<\/button>/);
  assert.match(stylesheet, /border-top:\s*1px dashed/);
  assert.match(editor, /className="editor-height-handle"/);
  assert.match(stylesheet, /cursor:\s*ns-resize/);
  assert.match(server, /spawnSync\("make"/);
  assert.match(server, /\["commit", "-m", message, "--", "main\.tex"\]/);
  assert.match(server, /listen\(PORT, "127\.0\.0\.1"/);
});
