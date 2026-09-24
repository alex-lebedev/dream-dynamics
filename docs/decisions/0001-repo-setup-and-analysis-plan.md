# ADR-0001 — Repo setup, scope, and analysis plan

- **Status:** accepted
- **Date:** 2026-07-18
- **Deciders:** Alex (owner); drafted with the research-engineer agent

## Context

`psychohistory` held two manuscripts and a pile of data (DreamSeer + DreamBank CSVs, a
Swiss-Ephemeris celestial spine, and five matched case-control event cohorts) with **no
git, no structure, no code**. Goal: turn it into a reproducible "insights machine" that
reads dreams as a signal of collective mood / societal dynamics, producing rigorous papers
and popular writeups. Profiling established DreamSeer as a dense, daily, **multilingual**
(65% EN / 20% RU / 15% other) population signal over ~2024-03 → 2026-06 (≈30k reports, 5.5k
users), with a steep adoption trend and healthy user spread.

## Decisions

1. **Governance model** — adopt the `scdnb` doctrine adapted for research: file-over-app,
   numbered knowledge dirs + conventional `src/` code, a `CLAUDE.md` constitution,
   `docs/` standards, `50-wiki/` regenerable layer, `.claude/` skills + a methodology-critic
   subagent, and per-dataset data-cards.
2. **Celestial/ephemeris — kept for a late stage** (revised from "exclude"). It stays in
   `10-data/processed/celestial/` and is **excluded from the primary pipeline**; reserved
   for a late-stage, pre-registered, negative-control-style test of dreams ↔ celestial
   dynamics. Seasonality control in main analyses uses plain calendar features.
3. **Flagship analysis — event-study** (matched local controls; reuses the event *cohorts*,
   but control matches are **re-derived** for the dream outcome, since the existing ones were
   celestial-matched). Salience-screened events; English cohort first, then Russian as a
   built-in replication + cross-cultural (socionomic-distance) test.
4. **Privacy — private remote**; raw dream data gitignored and never pushed; only
   de-identified aggregates tracked. Secrets via env/secrets-manager. (See `docs/ETHICS.md`.)
5. **External data — maximal**: DreamBank + Hall/VdC annotations (measurement validation);
   Reddit r/Dreams & r/Nightmares via Academic Torrents per-subreddit files (independent
   replication, 2016→2025); SDDb; and the published Šćepanović/Barrett/Quercia + Mallett
   corpora as positive controls.

## Consequences

- The primary spine is DreamSeer; all other corpora are **validation**, not pooled into the
  main signal.
- Event-study is favored because *local* comparisons are robust to the adoption trend that
  would contaminate long-run coupling.
- Language stratification is mandatory (multilingual population); enables a cross-cultural
  specificity design.
- Need to build a dream-appropriate control-matcher (not the celestial one).
- git 2.28.0 is the local toolchain; `op` (1Password CLI) and a torrent client are not
  installed on this machine — the Reddit pull and secret-backed loaders are documented in
  `RUNBOOK.md` for the environment that has them.

## Follow-ups

- ADR-0002: pre-registration of the flagship event-study (outcomes, windows, events).
- ADR-0003: DVC (or LFS) remote for large processed binaries + celestial parquet.
