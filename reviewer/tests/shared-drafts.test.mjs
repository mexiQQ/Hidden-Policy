import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, resolve } from "node:path";
import test, { after, before } from "node:test";
import { fileURLToPath } from "node:url";

const reviewerDir = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const port = 44_000 + (process.pid % 1_000);
const api = `http://127.0.0.1:${port}/api`;
let server;
let draftDir;
let draftPath;

async function waitForServer() {
  for (let attempt = 0; attempt < 50; attempt++) {
    try {
      const response = await fetch(`${api}/health`);
      if (response.ok) return;
    } catch {
      // The child process may still be binding its port.
    }
    await new Promise((resolveWait) => setTimeout(resolveWait, 50));
  }
  throw new Error("Draft test server did not start");
}

before(async () => {
  draftDir = await mkdtemp(resolve(tmpdir(), "hidden-policy-drafts-"));
  draftPath = resolve(draftDir, "drafts.json");
  server = spawn(process.execPath, [resolve(reviewerDir, "local-server.mjs")], {
    cwd: reviewerDir,
    env: { ...process.env, REVIEWER_API_PORT: String(port), REVIEWER_DRAFT_PATH: draftPath },
    stdio: "ignore",
  });
  await waitForServer();
});

after(async () => {
  server?.kill("SIGTERM");
  await rm(draftDir, { recursive: true, force: true });
});

test("loads the manuscript after it moved into paper", async () => {
  const response = await fetch(`${api}/document`);
  assert.equal(response.status, 200);
  const document = await response.json();
  assert.equal(document.title, "Hidden Policies: The Risk Behind Frontier Risks");
  assert.ok(document.blocks.length > 0);
  assert.equal(document.blocks[0].kind, "title");
});

test("persists browser-shared drafts to a local file and clears the file", async () => {
  const emptyResponse = await fetch(`${api}/drafts`);
  assert.deepEqual((await emptyResponse.json()).drafts, []);

  const payload = {
    version: 1,
    head: "test-head",
    savedAt: "ignored-client-time",
    drafts: [{
      blockId: "block-1", originalRaw: "Original", kind: "paragraph", label: "test paragraph", startLine: 12,
      proposal: { updatedRaw: "Updated", newText: "Updated" },
    }],
  };
  const saveResponse = await fetch(`${api}/drafts`, {
    method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(payload),
  });
  assert.equal(saveResponse.status, 200);
  assert.equal(JSON.parse(await readFile(draftPath, "utf8")).drafts[0].proposal.updatedRaw, "Updated");

  const sharedResponse = await fetch(`${api}/drafts`);
  assert.equal((await sharedResponse.json()).drafts[0].blockId, "block-1");

  const clearResponse = await fetch(`${api}/drafts`, {
    method: "POST", headers: { "content-type": "application/json" },
    body: JSON.stringify({ version: 1, head: "test-head", savedAt: "", drafts: [] }),
  });
  assert.equal(clearResponse.status, 200);
  const clearedResponse = await fetch(`${api}/drafts`);
  assert.deepEqual((await clearedResponse.json()).drafts, []);
});
