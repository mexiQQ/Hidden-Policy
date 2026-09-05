# Manuscript

This directory contains the paper source:

- `main.tex`: manuscript entry point;
- `math_commands.tex`: shared notation and macros;
- `references.bib`: bibliography;
- `iclr2027_conference.{sty,bst}`: official, locally downloaded ICLR files.

Build from the repository root with `make`. Run `make template` first when the
ICLR files are absent, and `make clean` to remove generated LaTeX artifacts.
The template files, PDF, logs, and auxiliary build outputs are intentionally
ignored by Git.
