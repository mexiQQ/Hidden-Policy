# Hidden Policy paper reviewer

A local, ICLR-styled tracked-changes editor for `../main.tex`.

## Start

On macOS, double-click `start-reviewer.command`, or run:

```bash
cd reviewer
pnpm install
pnpm review
```

Open <http://localhost:3000>.

## Review workflow

1. Click an editable paragraph or formula to open the docked revision editor on the right.
2. Keep the green/pink tracked diff visible on the paper while you continue editing; paragraph highlights update as you type.
3. Press Command/Ctrl+Enter or click **Update draft** to finish editing.
4. Insertions appear on green; deletions appear on pink with a strike-through.
5. Click **✓** to rebuild the PDF and commit only `main.tex`.
6. Click **×** to discard that proposed change and restore the original text.

Drag the panel's left grip to change its width. Drag the horizontal grip below the text or LaTeX formula editor up and down to change its height. The grip also supports the Up and Down arrow keys.

Temporary paragraph and formula drafts are autosaved to the local `reviewer/.review-drafts.json` file as you type. The in-app browser and Chrome share these drafts when they use the same local review server; refresh the other browser to load the latest version. Reloading the page or restarting the server restores matching drafts as pending tracked changes. Existing browser-only drafts are migrated automatically the first time the shared store is empty. Accepting or rejecting a change removes its saved draft; stale drafts are restored only when their original source block still matches the manuscript.

The local API binds only to `127.0.0.1`, accepts requests only from the local review page, validates the LaTeX build before committing, and restores the original source if the build fails.

Inline citations, references, and emphasis are protected while editing so their LaTeX markup remains intact. Inline and display formulas have their own LaTeX editor and live mathematical preview. ICLR section numbering, theorem bodies, quotations, numbered lists, equation numbers, and cross-references are rendered as structured paper elements. Complex non-math environments remain read-only.
