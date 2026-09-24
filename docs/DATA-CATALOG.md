# DATA CATALOG

Every dataset the project uses, its role, provenance, license, and status. Each dataset
also gets a machine-readable data-card in `10-data/manifests/`. **Nothing enters an analysis
without a data-card.**

Legend — Status: ✅ present locally · ⬇️ to download (see `RUNBOOK.md`) · 🧱 built elsewhere
(outputs only present).

---

## Dreams

| Dataset | Role | Coverage | Source / access | License | Sensitivity | Status |
|---|---|---|---|---|---|---|
| **DreamSeer** | **PRIMARY signal** | daily, ~2024-03→2026-06 (dense); 30,267 reports; 5,526 users; EN/RU/other | Alex's app export | private | **S3** | ✅ `10-data/raw/dreamseer/` |
| **DreamBank** | normative baseline; method calibration | individual longitudinal (dates unreliable) | local; also HF `DReAMy-lib`, Zenodo `18159468` | research; cite Domhoff & Schneider 2008 | S2 | ✅ `10-data/raw/dreambank/` |
| **DreamBank + Hall/Van de Castle annotations** | **measurement validation** (model scores vs gold coding) | 27,952 reports | HF `gustavecortal/DreamBank-annotated` | research | S1/S2 | ⬇️ |
| **Reddit r/Dreams + r/Nightmares** | **independent replication**; extends to 2016 | 2016→2025-12, timestamped | Academic Torrents (Watchful1 per-subreddit files) + `Watchful1/PushshiftDumps` (MIT) | research; don't re-host | S2/S3 | ⬇️ |
| **Šćepanović/Barrett/Quercia Reddit corpus** | **positive control** (COVID + Ukraine effects) | 185k posts, 2016–2022 | Figshare `10.6084/m9.figshare.23618064` | research | S2 | ⬇️ |
| **Mallett "dysphoric dreaming"** | positive control (COVID interrupted-time-series) | r/Dreams + r/news, 2019–2020 | Zenodo `10.5281/zenodo.18940544` | research | S2 | ⬇️ |
| **SDDb (Sleep & Dream Database)** | baseline / robustness | 44,500+ reports | Zenodo `10.5281/zenodo.11662063` (Bulkeley) | research; cite Bulkeley | S2 | ⬇️ |

## Societal signals (daily predictors)

| Signal | Indexes | Cadence | Source | License | Status |
|---|---|---|---|---|---|
| **GDELT / Geotone** | global news tone + theme/country attention | daily (15-min) | HF `davidcummings/geotone-global-news-signals`; GDELT GKG | CC BY 4.0 | ⬇️ |
| **VIX + S&P 500** | market "fear"; socionomic mood | daily (trading) | FRED (`VIXCLS`, `SP500`) | public | ⬇️ (no key) |
| **Daily EPU** | economic-policy uncertainty | daily | policyuncertainty.com | public | ⬇️ |
| **SF Fed Daily News Sentiment** | macro news sentiment | daily | FRBSF | public | ⬇️ |
| **Hedonometer** | Twitter collective happiness | daily (≤~2023) | hedonometer.org API | CC BY-NC-SA | ⬇️ |
| **Wikipedia pageviews** | collective attention | daily | Wikimedia REST API | CC0 | ⬇️ (no key) |
| **Google Trends** | public worry/attention | daily/weekly | `pytrends` | ToS (sampled) | ⬇️ |
| **ACLED** | conflict / protest events | daily/event | ACLED API | academic (key) | ⬇️ (needs key) |

## Events (systematic cohorts — outputs already built; scripts ran elsewhere)

| Cohort | Source | N (core) | Notes | Status |
|---|---|---|---|---|
| Earthquakes | USGS ComCat M5.5+ 1990→2026 | 17,844 | matched case-control scaffold | 🧱 `10-data/processed/events/` |
| Aviation accidents | NTSB `avall.mdb` | 26,246 | fatal/hull-loss cohorts | 🧱 |
| Terror attacks | GTD (nkill≥20) 1970→ | 2,570 | severe-incident cohorts | 🧱 |
| Coups (political shocks) | Powell & Thyne | 249 | successful coups 1950→ | 🧱 |
| Interstate conflict onsets | UCDP v25.1 | 109 | new-conflict onsets | 🧱 |

> ⚠️ The `*_control_pool` / `*_control_matches` in these outputs were matched against the
> **celestial** spine. For dream-outcome event-studies we **re-derive** controls on
> dream-sampling confounds (weekday, season, trend, volume). Only the `*_core` event tables
> (dates, magnitude, fatalities, geo) are reused directly.

## Celestial (kept for a late stage — NOT in the primary pipeline)

Swiss-Ephemeris (`swisseph 2.10.03`) daily spine (1,077,480 rows, yr −400→2550) + derived
features (lunations 19,790; ingresses 167,413; retrograde stations 199,365) + EOP/Δt. Lives
in `10-data/processed/celestial/`. Reserved for the pre-registered celestial exploration
(ADR-0001 §2). See `metadata_build.json` for full provenance + QA.

---

## DreamSeer schema (42 columns) & measured profile (2026-07-18)

**Identifiers/time:** `documentID`, `userID`, `createdAt` (date).
**Text:** `text` (raw dream), `interpretation` (LLM; 96.7% coverage), `chatSummary` (8.9% —
too sparse for time series), `themes` (keyword list; 96.7%), `detected_symbolism` (50.5%).
**Embedding:** `X`,`Y`,`Z` (+ `coordinates` dict; 97.8%).
**Feedback:** `rating` (0–5; 13% — not for time series).
**Affect — Plutchik-8 (0–1; ~99.7% coverage):** `fear` (μ .36), `anger` (.25), `sadness`
(.29), `disgust` (.18), `joy` (.12), `trust` (.33), `anticipation` (.30), `surprise` (.70).
**Content tags (0–1; ~99.7%):** `danger` (.72), `family` (.52), `health`, `weapon`,
`animal`, `flying`, `swimming`, `place_house`, `place_nature`, `indoors`, `outdoors`,
`romantic`, `social`, `friend`, `stranger`, `situation_childhood`, `game`, `music`, `food`,
`object_inanimate`, `object_animate`.

**Measured facts driving the design:**
- **Language:** 65.5% ASCII/EN, **19.5% Cyrillic/RU**, 15.0% other non-ASCII → stratify.
- **Volume:** grows ~100/mo (late-2023) → ~1,500–1,900/mo (2025-26); dense from **2025-05**.
- **Users:** healthy spread (top-1 user 1.5%, top-10 7.0%; 725 users ≥10 dreams).
- **Text:** median 296 chars / 58 words.
- **Junk dates:** 26 pre-2023 rows (drop). `createdAt` max 2026-07-09 (export cutoff).
- Dreams skew negative (fear ≫ joy) — expected for dream content.
