# Measuring a dreaming population

Figures, videos, de-identified aggregates and the code that drew them, for *Measuring a dreaming population: multilevel temporal organization in 75,000 dream reports*.

- **Start here:** [`index.html`](index.html) — the illustrated account, with the videos
  ([read it rendered](https://lebedevlabs.com/dream-dynamics/))
- **The manuscript:** [`manuscript.html`](manuscript.html) (citations resolved) or the source,
  [`dream-dynamics.md`](70-papers/academic/dream-dynamics.md)
- **Videos:** [`60-results/videos/`](60-results/videos/)

## What this is

A population of people wrote down their dreams in an app, daily, for two years — 30,241 reports
from 5,524 users. Set beside roughly 45,000 more from three public archives, that is the corpus. This
repository holds what could be measured in it without ever publishing one of those reports: the
emotional shape of a dream report over its own length, the geometry of the language it is written
in, and a set of tests for whether any of it tracks the news.

Some of those tests came back positive and some came back null. The null results are in the paper
and in the article, in the same detail as the positive ones.

## What is here, and what is not

This is a **curated showcase**, not a mirror of the working repository. It ships the importable
package (`src/psychohistory/`), the 45 analysis scripts cited in the manuscript's reproducibility table (including those that draw every figure and video it ships),
the aggregate tables those scripts read, the data-cards describing each source, the methods and
ethics documents, and the manuscript.

It does **not** contain dream reports, and it never will. The primary corpus is personal narrative
written by identifiable people; raw text and user identifiers do not leave the machine they were
analysed on. No cell published anywhere in this repository rests on fewer than
20 reports or 5 contributors — see [`docs/ETHICS.md`](docs/ETHICS.md), and
[`scripts/check_release_cells.py`](scripts/check_release_cells.py), which is the check that enforces
it and which refuses to build this release if it fails.

Also absent: the other 60 analysis scripts in the working repository (exploratory passes
and superseded drafts), the corpus-download and feature-build tools, and the working notes.

## What you can and cannot rebuild

Being exact about this, because "analysis code" in a repository with no data usually means less than
it sounds like:

| | |
|---|---|
| **Runs as shipped** | `2026-08-22-05-chain-figure.py` — the measurement-chain diagram draws from nothing but itself. |
| **Runs from shipped tables** | The panels of `fig1`, `fig2_geometry` and `fig4_arrow_robustness` that read the aggregates in `60-results/showcase/*.csv`. Those CSVs are the numbers behind the claims; you can check the figures against them, and check them against the paper. |
| **Does not run** | Everything else. The corpus figures need the raw reports; the videos need cached sentence-level embeddings and sentiment (tens of GB, derived from raw text). They are shipped to be *read* — so that what was computed is inspectable — not to be executed. |

For the external corpora, go to the sources rather than to us: DreamBank, the Sleep and Dream
Database, and the published Reddit dream corpora. Each is described with its licence and retrieval
terms in [`10-data/manifests/`](10-data/manifests/).

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[signals,external,dev]"
```

## Licence and citation

Code is MIT ([`LICENSE`](LICENSE)); prose, figures and videos are CC BY 4.0
([`LICENSE-CONTENT.md`](LICENSE-CONTENT.md)). Cite the manuscript for the findings, and each data
source on the terms in its own data-card.

Built by `scripts/build_public_release.py` from a private research repository; the file list is in
[`RELEASE-MANIFEST.txt`](RELEASE-MANIFEST.txt).
