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

1. Hover over a sentence to highlight it, then click that sentence or a formula to open the docked revision editor on the right.
2. Edit one sentence at a time while the green/pink tracked diff remains visible on the paper.
3. Select text and click **B**, or press Command/Ctrl+B, to add or remove LaTeX bold formatting.
4. Click **Copy sentence** to copy the current sentence as LaTeX, or **Copy paragraph** to copy the complete paragraph with all current draft edits. Formula editors copy their current formula source.
5. Changes are saved automatically as you type; press Escape or close the panel when you are done editing.
6. Insertions and added bold formatting appear on green; deletions appear on pink with a strike-through.
7. Use **✓ Accept & commit** in the Revision Editor to rebuild the PDF and commit the selected sentence change to `main.tex`.
8. Use **× Reject** in the Revision Editor to discard the selected sentence change and restore its original text.

Drag the panel's left grip to change its width. Drag the horizontal grip below the text or LaTeX formula editor up and down to change its height. The grip also supports the Up and Down arrow keys.

Temporary sentence and formula drafts are autosaved to the local `reviewer/.review-drafts.json` file as you type. The in-app browser and Chrome share these drafts when they use the same local review server; refresh the other browser to load the latest version. Reloading the page or restarting the server restores matching drafts as pending tracked changes. Existing browser-only drafts are migrated automatically the first time the shared store is empty. Accepting or rejecting a change removes its saved draft; stale drafts are restored only when their original source block still matches the manuscript.

The local API binds only to `127.0.0.1`, accepts requests only from the local review page, validates the LaTeX build before committing, and restores the original source if the build fails.

Inline citations, references, and emphasis are protected while editing so their LaTeX markup remains intact. Inline and display formulas have their own LaTeX editor and live mathematical preview. ICLR section numbering, theorem bodies, quotations, numbered lists, equation numbers, and cross-references are rendered as structured paper elements. Complex non-math environments remain read-only.
