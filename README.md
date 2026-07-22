# The physics of dreams — analysis code

Analysis code accompanying the manuscript *The physics of dreams: scaling laws, low-dimensional geometry and
an emotional arrow of time in a dreaming population.*

This repository contains **only the analysis scripts** (and the small importable package they call). It
contains **no data and no figures**: raw dream text and user identifiers are private and are never shared,
and even de-identified aggregate outputs are produced locally when the scripts are run.

## What the analyses do

A dense, daily, multilingual stream of ~30,000 dream reports (DreamSeer; 30,241 cleaned reports from 5,524
users, 2024–2026) — with DreamBank, Reddit r/Dreams and the Sleep & Dream Database — is analyzed as a complex
system under one multilingual sentence-embedding + sentiment instrument, with every claim tested against
explicit nulls (feature shuffles, sentence-order shuffles, a Gaussian reference, a stable-user panel, and
autocorrelation-preserving surrogates). Headline results, honestly tiered:

- **Robust:** classical scaling laws (Zipf ≈1.15, Heaps V∼N^0.57); a low-dimensional manifold (two-NN
  intrinsic dimension ≈25 vs 112/131 nulls in 384-d); a single dense percolation "continent" (95% connected
  at cosine 0.5 vs 0% shuffled); a demographic sex axis (Cohen's d ≈ 0.22, verbosity-robust).
- **Cross-corpus replicated but exploratory:** an emotional arrow of time — dreams end darker across five
  corpus-language samples (AUC 0.647–0.667), with an LLM-interpretation comparator ending lighter.
- **Frontier:** a universal descending *mean* arc (cross-corpus |cos| 0.957 vs 0.25 shuffled); critical
  slowing down before mood-darkenings (Kendall τ = +0.236, p = .005; 12 episodes).

Aggregate dream affect does **not** detectably track external market/news/cultural indicators — reported only
as a one-paragraph specificity control (the structure is intrinsic).

## Contents

```
analyses/            the 12 committed analysis scripts (one per result component)
src/psychohistory/   the importable package the scripts call (trimmed to only what they import)
pyproject.toml       package metadata (so the scripts can import psychohistory)
```

`analyses/` maps one script per result component: physics/scaling & intrinsic dimension, collective dynamics
(percolation, stable-panel geometry, critical slowing down), within-dream narrative physics, cross-corpus
arrow/arcs, emotional-arc basis, demographics, the individual-signature replication, the two societal-signal
specificity-control scripts, and the two figure-rendering scripts. Each script documents its outputs in its
docstring and writes de-identified aggregate results to a local `60-results/` folder when run.

## Running

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .                         # installs the psychohistory package
PYTHONPATH=src PYTHONNOUSERSITE=1 USE_TF=0 python3 analyses/<script>.py
```

The scripts read local corpora from a `10-data/` folder. Because the primary DreamSeer data are **private**,
the DreamSeer-dependent scripts cannot be run end-to-end from this repository alone; the code is published for
transparency and methodological review.

## Data & privacy

No raw dream text or user identifiers are included in, or reconstructable from, this repository. DreamSeer raw
data are private (app-owner licensed) and are never shared; only de-identified, aggregated derivatives are
ever produced, with a per-cell minimum of ≥ 20 reports / ≥ 5 users. DreamBank, Reddit r/Dreams and SDDb must
be obtained from their original providers under their terms. Any external release of results should follow an
institutional ethics/IRB determination and the relevant terms-of-service/licensing review.

## Citation

If you use this code, please cite the manuscript. `[AUTHOR TO CONFIRM: final author list, venue, year, DOI.]`

```bibtex
@article{dream-dynamics,
  title  = {The physics of dreams: scaling laws, low-dimensional geometry and an emotional arrow of time in a dreaming population},
  author = {Lebedev, Alexander V. and others},
  year   = {2026},
  note   = {Manuscript; venue and DOI to be confirmed}
}
```

## License

MIT — see [`LICENSE`](LICENSE). Third-party corpora and societal-signal sources retain their own licenses and
are not redistributed here.
