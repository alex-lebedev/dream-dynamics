# ETHICS & DATA GOVERNANCE

DreamSeer is human-subjects data: user identifiers plus intimate personal narratives.
This document governs how data is stored, transformed, shared, and published. It binds all
agents and all code. When it conflicts with convenience, it wins.

---

## 1. Data sensitivity tiers

| Tier | Examples | Rule |
|---|---|---|
| **S3 — sensitive** | DreamSeer `text`, `interpretation`, `userID`; raw Reddit text | **Never** committed, pushed, or shared. Local-only, under `10-data/raw/` (gitignored). |
| **S2 — restricted-license** | licensed third-party corpora (Reddit dumps, GTD, NTSB) | Stored under `10-data/{raw,external}/` (gitignored). Do not redistribute; store IDs + re-fetch where terms require. |
| **S1 — de-identified aggregate** | day/week feature tables (means, counts) | Tracked in git; safe to share. No user IDs, no verbatim text, respect min-N (§4). |
| **S0 — public** | societal signals (VIX, GDELT), event metadata, code, docs | Freely tracked/shared. |

## 2. Hard rules

1. **Raw dream text never leaves the machine.** Not in git, not in a push, not pasted into
   an external API without an explicit, logged decision that consent/ToS/IRB permits it.
2. **Default output is aggregate.** Pipelines emit day/week-level tables by default.
   Dream-level exports require de-identification (§3) and a recorded justification.
3. **De-identify before any individual-level sharing** (§3).
4. **Secrets** (Reddit / ACLED / GitHub / any API keys) live in environment variables or a
   secrets manager (1Password/GCP Secret Manager). Never hard-coded, never committed,
   never at rest in cleartext. Loaders read from `os.environ`.
5. **This repository has no public remote.** If pushed, the remote is private; `.gitignore`
   guarantees raw data cannot be added by `git add -A`. Code, de-identified aggregates, figures
   and prose may be published **only** through the curated export defined in
   `docs/decisions/0004-public-showcase-release.md`, and only once the §5 checklist below is
   closed. Making *this* repository public is not an available option, because that would turn
   every past commit and every future `git add` into a disclosure decision. Rule 1 is unaffected
   by this and remains absolute.

## 3. De-identification protocol (for any dream-level artifact)

- Mask person names, locations, organizations → `Person1`, `Location1`, `Org1`
  (e.g., `DReAMy` anonymizer or Microsoft Presidio).
- Strip/replace `userID` with a salted pseudonymous key held only locally.
- Remove free-text that could re-identify (rare place + date combinations).
- Manually spot-check a sample before sharing.

## 4. Aggregation / re-identification threshold

- Do not publish any cell (e.g., language × day) computed from **fewer than N=20** dreams
  or fewer than **5 distinct users**; suppress or coarsen (widen the window) instead.

## 5. Before publication (checklist — blocks external release, not internal building)

- [ ] Determine IRB status: exempt determination vs. review, per institution.
- [ ] Confirm DreamSeer Terms of Service / consent covers secondary research use.
- [ ] Confirm third-party dataset licenses permit the intended use + citation.
- [ ] De-identify any shared examples; verify min-N on every reported cell.
- [ ] Data-availability statement: share code + aggregates; point to original sources for
      restricted corpora (don't re-host).

## 6. Third-party licensing notes

- **Reddit dumps (Academic Torrents / Pushshift):** research use; do not re-publish raw
  text. Prefer sharing post IDs + derived aggregates.
- **DreamBank / SDDb:** cite Domhoff & Schneider (2008) and Bulkeley; respect site terms.
- **GTD, NTSB, UCDP, Powell–Thyne, USGS:** cite per each provider's requirement (see data-cards).
- **GDELT / Geotone:** CC BY 4.0 — cite "Geotone / GDELT".

## 7. Provenance

Every dataset has a data-card in `10-data/manifests/` recording source, license,
retrieval date, and sha256. No undocumented data enters an analysis.
