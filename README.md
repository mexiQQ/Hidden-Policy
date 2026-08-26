# Hidden Policies — ICLR 2027

Research draft for the Hidden Policy project, currently centered on two foundational questions:

1. **What is a Hidden Policy?**
2. **Why is it important?**
   - hidden policies as a cross-cutting mechanism behind frontier risks / preparedness failures;
   - hidden policies as a frontier risk in their own right through policy non-identifiability.

## Current title

**Hidden Policies: The Risk Behind Frontier Risks**

The title is intentionally ambitious. The current draft develops the conceptual spine; later work will add construction, diagnostics, mechanism/localization, and removal experiments.

## ICLR 2027 template

This repository is written for the **official ICLR 2027 LaTeX style**. The official author guidelines are:

- https://iclr.cc/Conferences/2027/AuthorGuidelines

The official style-file archive is:

- https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip

The paper expects the following official files at repository root:

- `iclr2027_conference.sty`
- `iclr2027_conference.bst`

Run `make template` to download the official archive and copy these files into the repository, or download the archive manually from ICLR and place the files at repository root.

**Do not replace the ICLR style with a third-party modified copy for submission.**

## Local build requirements

Recommended on macOS:

- **MacTeX** (full installation is easiest), or a sufficiently complete **TeX Live**;
- `latexmk`;
- `pdflatex`;
- `bibtex`;
- `curl` and `unzip` if using `make template`.

LaTeX packages used directly or by the ICLR style:

- `times`
- `natbib`
- `eso-pic`
- `fancyhdr`
- `hyperref`
- `url`
- `booktabs`
- `amsmath`
- `amssymb`
- `amsthm`
- `mathtools`
- `microtype`
- `xspace`

A full MacTeX installation contains these packages.

## Build

```bash
make template   # first time only, or whenever the official template changes
make
```

Equivalent direct build:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Manual build:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Clean generated files:

```bash
make clean
```

## Notation design

The draft deliberately keeps notation small and consistent:

- `h`: interaction history available to the model;
- `a`: model action;
- `π(a | h)`: effective model policy;
- `g(h) ∈ {0,1}`: condition recognizer / gate;
- `π_0`: reference behavioral regime;
- `π_H`: hidden behavioral regime;
- `𝓡`: interaction regime under which the observer sees the model;
- `𝓡'`: reachable revealing / counterfactual interaction regime;
- `P_π(τ | 𝓡)`: observable trajectory distribution;
- `[π]_𝓡`: policy equivalence class under an interaction regime.

The notation is **functional, not mechanistic**: writing `π_0` and `π_H` does not assume two localized subnetworks or two literally stored policies. The layer/localization question is deliberately deferred.

## Current conceptual commitments

- **Hiddenness is interaction-relative, not benchmark-relative.** Evaluation is only one interaction regime.
- **Policy is different from behavior.** A policy generates a distribution over realized actions/trajectories.
- **Policy is different from objective.** Hidden objectives are an objective-level concept rather than the definition of Hidden Policy.
- **Situational awareness is condition recognition.** It can be represented by a sophisticated gate `g(h)`.
- **Strategic reasoning is policy structure.** It can live inside the activated decision regime rather than being an extra external component.
- **Hidden Policy does not require determinism, intentional implantation, maliciousness, deception, or an explicit objective.**
- **Behavioral disappearance does not imply policy removal.** Condition invalidation, behavioral unreachability, policy suppression/substitution, and genuine policy removal are distinct claims.

## ICLR 2027-specific notes

The ICLR 2027 author guidelines currently specify a **9-page main-text limit at initial submission** (references excluded), double-blind review, and a **mandatory AI-use statement** that does not count toward the page limit. The draft contains an AI-use disclosure placeholder that should be edited to accurately reflect the final workflow before submission.

## Source / citation status

The current bibliography contains the foundational sources needed for the first two questions, including Sleeper Agents, Alignment Faking, Hidden Objectives, and frontier preparedness/safety frameworks. Before submission, framework versions, dates, URLs, and publication metadata should be re-verified against the final manuscript.
