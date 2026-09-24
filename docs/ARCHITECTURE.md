# ARCHITECTURE — the "why"

This document is the design rationale. `CLAUDE.md` is the "what to do"; this is "why it's
shaped this way." It adapts the `scdnb` second-brain doctrine to a **research/data-science**
repository whose job is to manufacture credible insights (papers + press) from dream data.

---

## 1. The thesis and the risk

**Thesis:** dreams are a noisy but real read-out of the collective unconscious/mood; when
aggregated over a population and over time they should co-move with societal dynamics and
respond to significant events. There is published precedent (Reddit dream studies around
COVID and the Ukraine war; the lineage socionomics papers).

**The risk:** this class of claim is catnip for spurious correlations, and reputational
downside is asymmetric. So the architecture is built to make the *rigor* mechanical and the
*credibility* the default — validation-first, confound-first, preregistered, replicated.

## 2. Core design principles

1. **Reproducibility is the product.** The asset is `raw/ + code`, from which every table,
   figure, and number regenerates. Notebooks explore; the package decides.
2. **File-over-app / plain-text durability** (from `scdnb`): CSV + markdown + code that read
   in 20 years; tooling (`.claude/`, even the model) is swappable.
3. **A one-directional data flow:** `raw → interim → processed`. Raw is immutable; anything
   downstream is disposable and rebuildable. This is what makes the science auditable.
4. **Primary vs. validation separation.** DreamSeer is the single spine; every other corpus
   exists to *challenge* a DreamSeer result, never to quietly pad it.
5. **Confounds and nulls are first-class** (see `docs/METHODS.md`). The pipeline emits the
   negative controls and sensitivity analyses alongside the headline number.
6. **Privacy by construction** (see `docs/ETHICS.md`): the default artifact is a
   de-identified aggregate; raw text cannot be `git add`-ed.

## 3. Why this structure (hybrid numbered + code)

`scdnb` numbers its knowledge domains for stable sort and agent navigability. A research
repo also needs an *importable code package* and reproducible-pipeline conventions
(à la Cookiecutter Data Science). We therefore split:

- **Numbered dirs = knowledge & artifacts** (`10-data`, `40-notes`, `50-wiki`, `60-results`,
  `70-papers`, `90-archive`) — sacred content, stable order.
- **Unnumbered dirs = code & tooling** (`src/`, `scripts/`, `analyses/`, `docs/`, `.claude/`,
  `templates/`) — Python-conventional, swappable.

This keeps the `scdnb` benefits (a regenerable `50-wiki/`, `docs/` rationale, `.claude/`
skills + critic subagent, per-item provenance) while respecting how data-science code wants
to be laid out (installable `psychohistory` package, `raw/interim/processed`).

## 4. The insights machine (the loop)

```
hypothesis (00-inbox / 40-notes)
   → pre-register (70-papers/academic/preregistrations, ADR)
   → build signal (src/psychohistory/signals) + dream features (…/dreams)
   → match / model (…/matching, …/stats)
   → methodology-critic subagent  ── JSON PASS / REVISE / FAIL (bounded loop)
   → finding (50-wiki/findings.md) + figures/tables (60-results)
   → external replication + positive-control reproduction (validate-external)
   → draft-paper (academic) + draft-popular (press release)
```

The `methodology-critic` is the independent gate (mirrors `scdnb`'s `clinical-critic`): it
checks confound handling, multiplicity, leakage, overclaiming, and validation — and it does
not rewrite, it returns a verdict. Nothing is a "finding" until it passes and replicates.

## 5. State & memory

- `50-wiki/index.md` — live state snapshot + active threads (read first).
- `50-wiki/log.md` — append-only dated changelog.
- `50-wiki/findings.md` — the compounding, regenerable ledger of what we believe and how
  strongly (with links to the analysis + result that support it).
- git history — the time machine; ADRs in `docs/decisions/` — why we chose what we chose.

## 6. Future-proofing

- Content (raw data cards, findings, papers) is plain text and survives any tool change.
- The `.claude/` layer is translatable to any agent framework (Cursor reads `AGENTS.md`).
- Large binaries (celestial + event parquet) are kept out of git and will be DVC-tracked
  (ADR-0003) so clones stay small and history stays clean.
- The celestial layer sits dormant but ready for a late-stage, clearly-fenced exploration.

## 7. Open engineering follow-ups

1. DVC/LFS remote for `processed/*.parquet` + celestial (ADR-0003).
2. Re-derive the build scripts for the event cohorts (only the parquet outputs exist here;
   the `scripts/build_*.py` referenced in `metadata_build.json` were run elsewhere).
3. CI eval gate for the `.claude` skills (as in `scdnb`).
