# processed/events

Pre-built **societal event cohorts** (earthquakes, aviation, terror, coups, interstate conflict
onsets) as matched case-control tables (`*_core`, `*_outcome_day`, `*_enriched_daily`,
`*_control_pool`, `*_control_matches`, `*_cohort`).

The `*.parquet` here are **gitignored** (large binaries; version via DVC — ADR-0003). Provenance +
QA for how they were built is in `10-data/manifests/metadata_build_extradata.json`.

**Only the `*_core` tables are used** for dream event-studies. The shipped `*_control_matches`
were matched on the celestial spine and are NOT reused — dream-appropriate controls are re-derived
by `psychohistory.matching.controls` (see docs/METHODS.md §3A).
