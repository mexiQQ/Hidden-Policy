#!/bin/zsh
set -e

reviewer_dir="${0:A:h}"
bundled_root="/Users/my/.cache/codex-runtimes/codex-primary-runtime/dependencies"

if command -v node >/dev/null 2>&1 && command -v pnpm >/dev/null 2>&1; then
  :
elif [ -x "$bundled_root/node/bin/node" ] && [ -x "$bundled_root/bin/fallback/pnpm" ]; then
  export PATH="$bundled_root/node/bin:$bundled_root/bin/fallback:$PATH"
else
  echo "Node.js and pnpm are required. Install them, then run this launcher again."
  read "?Press Return to close…"
  exit 1
fi

cd "$reviewer_dir"
pnpm install --frozen-lockfile
echo ""
echo "Paper reviewer: http://localhost:3000"
echo "Keep this window open while editing. Press Control-C to stop."
echo ""
pnpm review
