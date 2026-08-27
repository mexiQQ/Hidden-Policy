import { spawn } from "node:child_process";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const bin = resolve(root, "node_modules", ".bin", "vinext");
const children = [
  spawn(process.execPath, [resolve(root, "local-server.mjs")], { cwd: root, stdio: "inherit" }),
  spawn(bin, ["dev"], { cwd: root, stdio: "inherit", env: { ...process.env, WRANGLER_LOG_PATH: ".wrangler/wrangler.log" } }),
];

let stopping = false;
function stop(signal = "SIGTERM") {
  if (stopping) return;
  stopping = true;
  for (const child of children) if (!child.killed) child.kill(signal);
}

for (const signal of ["SIGINT", "SIGTERM"]) process.on(signal, () => stop(signal));
for (const child of children) child.on("exit", (code) => {
  if (!stopping && code) { stop(); process.exitCode = code; }
});
