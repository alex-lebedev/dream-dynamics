# RUNBOOK — how to operate `psychohistory`

Everything you need to reproduce, extend, and analyze. Read `CLAUDE.md` + `docs/METHODS.md` +
`docs/ETHICS.md` first. Commands assume repo root and `PYTHONPATH=src` (or `pip install -e .`).

---

## 0. What already exists (done in the setup session)

- Repo scaffolded; git on `main`; `.gitignore` guarantees raw dream data is never committed.
- **DreamSeer signal built**: `10-data/processed/dreams/dreamseer_{daily,weekly}.csv` (+ gitignored
  per-dream parquet), QA report + anomalous-days + figure in `60-results/`.
- **Signals fetched**: `10-data/external/signals/` (VIX, S&P 500, Geotone tone/attention).
- **Analysis engine** (`src/psychohistory/`) implemented + smoke-tested: features, signals, matching
  (events + dream controls), stats (event-study, coupling, inference).
- **Grounding papers** in `40-notes/sources/`. `.claude/` skills + methodology-critic + templates in place.

## 1. Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[signals,external,dev]"     # adds langdetect, pytrends, datasets, etc.
# (a conda base with pandas/pyarrow/statsmodels also works via PYTHONPATH=src)
```

`langdetect` sharpens the EN/other split (Russian is script-detected regardless). Without it the
pipeline still runs on the built-in heuristic.

## 2. Secrets (never commit — docs/ETHICS.md)

```bash
export ACLED_KEY=...  ACLED_EMAIL=...        # ACLED conflict data
export GH_TOKEN=...                          # or: gh auth login
# store durably in a manager (1Password `op`, GCP Secret Manager); loaders read os.environ
```

## 3. Data acquisition — "what to download"

Idempotent; re-runnable. Roles/licenses in `docs/DATA-CATALOG.md`.

```bash
bash scripts/download_all.sh                 # runs everything below, best-effort
```
…or piecewise:
```bash
# grounding papers -> 40-notes/sources/
bash scripts/fetch_articles.sh

# daily societal signals (no key)
PYTHONPATH=src python -m psychohistory.signals.build --fred VIXCLS SP500 DFF UNRATE --geotone
#   EPU (daily):  curl -fL -o 10-data/external/signals/epu_daily.csv \
#                 https://www.policyuncertainty.com/media/All_Daily_Policy_Data.csv
#   Google Trends: pip install pytrends; then a small fetch of terms
#                  (war, anxiety, layoffs, insomnia, recession, nuclear) -> external/signals/
#   ACLED:        needs ACLED_KEY/ACLED_EMAIL (conflict/protest events)

# open dream corpora (Zenodo/Figshare): SDDb, Mallett, Scepanovic, DreamBank archive
PYTHONPATH=src python scripts/fetch_open_datasets.py --all

# DreamBank + Hall/Van de Castle annotations (measurement validation)
PYTHONPATH=src python scripts/fetch_dreambank_hvdc.py

# Reddit r/Dreams + r/Nightmares (independent replication) — needs a torrent client
#   see scripts/fetch_reddit_dreams.sh (per-subreddit Academic Torrents files, then extract)
```

> Put the raw DreamSeer + DreamBank exports at `10-data/raw/dreamseer/dreamseer_data.csv` and
> `10-data/raw/dreambank/dreambank_data.csv` (already there in this machine).

## 4. Build the primary signal

```bash
PYTHONPATH=src python -m psychohistory.dreams.build_features
```

## 5. Analyze — "what to do"

Flagship **event-study** (EN first, then RU as replication):
```python
PYTHONPATH=src python - <<'PY'
import pandas as pd
from psychohistory.matching.events import load_events
from psychohistory.stats.event_study import event_study
daily = pd.read_csv("10-data/processed/dreams/dreamseer_daily.csv", parse_dates=["date"])
en = daily[daily.lang == "en"]
# TODO: replace with a SALIENCE-SCREENED, in-window event set (see §7); magnitude alone is not salience.
ev = load_events(families=["terror"], start="2024-03-01", end="2026-06-15")
for oc in ["fear_mean", "nightmare_index_mean", "negativity_mean", "food_mean"]:  # last = negative control
    print(oc, event_study(en, ev.date.tolist(), oc, n_perm=5000))
PY
```
**Coupling** (exploratory unless preregistered):
```bash
PYTHONPATH=src python analyses/2026-07-18-01-coupling-demo.py
```
Then: write claims to `50-wiki/findings.md`, invoke the **methodology-critic** subagent, and run
`validate-external` before calling anything a finding.

## 6. Build the Reddit replication signal (after §3 Reddit pull)

Reuse the DreamSeer feature logic on Reddit text (map r/Dreams post → dream text; score emotions
with the same model or DReAMy) → daily/weekly features → re-run §5 designs → concordance check.

## 7. Immediate next tasks (priority order)

1. **Salience-screened event set (2024-06→2026)** for the flagship: curate high-salience shocks
   (major attacks, escalations, disasters, big political/economic shocks) relevant to EN and (separately)
   RU cohorts; or weight the systematic cohorts by GDELT coverage. Preregister outcomes/windows (ADR-0002).
2. **External data**: run the Reddit pull + `fetch_open_datasets.py --all`; reproduce the COVID
   (Mallett) and Ukraine-onset (Šćepanović) positive controls in our pipeline.
3. **Deconfound coupling**: detrend adoption + weekday/seasonal; re-test VIX→fear lead-lag out-of-sample.
4. **Measurement validation** vs Hall/Van de Castle (dreambank annotated).
5. **Private remote** (§8) + `langdetect` install + DVC for big parquet (ADR-0003).

## 8. Private remote (raw stays out)

```bash
git add -A && git commit -m "…"
gh repo create <org>/psychohistory --private --source=. --remote=origin --push
# .gitignore already excludes 10-data/raw|external|interim and *.parquet — verify nothing sensitive:
git ls-files | grep -E '10-data/(raw|external|interim)|\.parquet$' || echo "clean: no sensitive/binary tracked"
```
