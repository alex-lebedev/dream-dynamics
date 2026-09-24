# METHODS — Standing Statistical SOP

This is the methodological contract for every confirmatory analysis in this repo. It
exists because the topic ("dreams predict society") is exactly the kind of claim that
attracts both excitement and (justified) skepticism. Credibility is earned by handling
confounds, pre-registering, replicating, and reporting nulls. Inherits the rigor of the
lineage papers (matched confound adjustment; linear + mutual-information effect sizes;
time-lagged cross-correlation; socionomic/temporal-distance specificity; alternative-metric
replication; OSF preregistration).

---

## 0. Units of analysis

- **Dream-level**: one dream report (features + language + user + timestamp).
- **Day-level**: aggregate of dreams on a calendar day (the primary signal).
- **Week-level**: ISO-week aggregate (default for coupling analyses; more robust to
  day-to-day noise and weekday effects).
- Aggregation is **language-stratified**: EN / RU / other, plus an "all" pooled series.

## 1. Dream features (built by `psychohistory.dreams`)

- **Affect (Plutchik-8):** fear, anger, sadness, disgust, joy, trust, anticipation,
  surprise (per-dream 0–1; day = mean and share-above-threshold).
- **Content tags:** danger, weapon, family, flying, animal, health, water/swimming,
  childhood, stranger, … (day = mean / prevalence).
- **Derived indices:** `nightmare_index` (composite of fear+danger+sadness − joy,
  pre-registered weights), `negativity = mean(neg emotions) − mean(pos emotions)`.
- **Embedding:** day-level centroid; `drift` = distance from trailing 28-day centroid;
  `dispersion` = mean pairwise distance (collective coherence).
- **Novelty/entropy:** theme-distribution entropy and KL vs trailing baseline.
- **Volume controls:** n_dreams, n_users, mean text length, share new users.

Every feature is defined once, in code, with a pre-registered formula. No post-hoc
feature invention in confirmatory analyses.

## 2. Confounds — adjust for ALL of these before any claim

| Confound | Why it matters | Handling |
|---|---|---|
| **Adoption trend** | app grows ~15× over window; who joins changes | detrend (spline/STL); use *local* control windows; relative (z within trailing window) measures |
| **User-composition drift** | new cohorts differ in geography/style | mixed-effects with user random intercepts; within-user change models; stable-panel reweighting |
| **Day-of-week** | reporting + content vary by weekday | weekday fixed effects |
| **Seasonality** | annual mood/content cycles | calendar harmonics (sin/cos day-of-year), month FE — **NOT ephemeris** |
| **Sample size / day** | precision varies; small days noisy | inverse-variance weighting; min-N thresholds |
| **Language/geography** | multilingual, non-US population | stratify EN/RU/other; never pool across languages for a language-specific event |
| **Text length** | longer dreams score differently | include as covariate |
| **Multiplicity** | ~30 emotion/theme outcomes × many events/lags | Benjamini–Hochberg FDR within the outcome family; pre-register primary outcomes |

## 3. Designs

### A. Event-study / case-crossover  *(flagship)*
For each event (systematic cohort or curated shock) inside the dense window:
- Define an **exposure window** (e.g., days 0..+7) and **matched local control windows**
  (same weekdays/season, no other major event) — **re-derived for the dream outcome**
  (the existing `control_matches` were celestial-matched and are not reused as-is).
- **Salience weighting/screening**: only test events plausibly in the cohort's collective
  awareness (fatalities × GDELT media coverage; language-appropriate).
- Estimate change in each pre-registered outcome (Δ vs control) with a mixed model;
  inference by **permutation** (shuffle event/control labels) and **block bootstrap**.
- Primary contrast: negative-affect indices (fear, nightmare_index). Report effect size
  + CI, not just p.

### B. Continuous coupling
- Weekly dream vector vs weekly societal vector (VIX, GDELT tone, EPU, news sentiment,
  Hedonometer[≤2023], Wikipedia/Trends attention).
- **Lagged cross-correlation** and **VAR/Granger** (lead–lag: do dreams lag, co-move, or
  lead?); **mutual information** for non-linear coupling (lineage method).
- Inference on autocorrelated series: **circular-shift permutation** + Newey–West SE;
  prewhiten before cross-correlation.

### C. Topic / embedding drift
- Dream topic mixture (embedding clusters / themes) vs GDELT GKG theme mixture over time;
  measure co-movement of topic shares (e.g., violence/health/travel).

### D. Predictive ("leading indicator")  *(high value, high risk)*
- Does last week's dream affect improve **out-of-sample** nowcast/forecast of next week's
  societal indicator beyond an autoregressive baseline?
- **Purged, embargoed** walk-forward CV; strictly out-of-sample; no look-ahead in features.

## 4. Inference & specificity (non-negotiable)

1. **Pre-registration** (OSF) of primary hypotheses, outcomes, windows, and models before
   confirmatory runs. Exploratory work is labeled exploratory.
2. **FDR** (Benjamini–Hochberg) across the emotion/theme family and across lags/events.
3. **Resampling inference** for autocorrelation: block/stationary bootstrap; permutation.
4. **Alternative-metric replication** (signature move): any societal effect must hold
   across ≥2 independent indicators (e.g., VIX *and* GDELT tone).
5. **Negative controls** (turn skepticism into evidence of specificity):
   - date-shuffled / phase-randomized dream series (effect should vanish);
   - unrelated outcome (e.g., "food" tag should not respond to a terror attack);
   - **ephemeris placebo predictors** (retrograde/aspect flags) — reserved for the
     late-stage celestial exploration; if they "predict" as strongly as real signals, the
     pipeline is leaking.
6. **Sensitivity analyses**: window widths, aggregation level (day/week), with/without
   power users, EN-only vs pooled, alternative detrending.

## 5. Validation protocol (before anything is a "finding")

- **External replication:** re-run the same event-study on **Reddit r/Dreams/Nightmares**
  (independent platform/population). Concordant sign+significance = real.
- **Positive controls:** reproduce published effects in our pipeline — COVID-onset
  dysphoric-dream rise (Mallett interrupted time series) and the Feb-2022 Ukraine-onset
  violent-content shift (Šćepanović/Barrett/Quercia). If we can recover known effects, the
  instrument works.
- **Measurement validation:** correlate DreamSeer's model-scored emotions/tags against
  gold **Hall/Van de Castle** codings on the annotated DreamBank subset; report agreement.

## 6. Causal humility

Observational. Use explicit DAGs; prefer "consistent with / tracks / precedes" over
"causes." Event-study + negative controls + out-of-sample prediction are the strongest
available evidence short of intervention — frame accordingly.

## 7. Reporting standard

Every confirmatory result reports: N (dreams/days/users), effect size + CI, resampling p,
FDR status, the confound set adjusted for, the negative-control outcome, and the
replication result. A pre-registered null is published as a null.
