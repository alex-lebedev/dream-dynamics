# VALIDATION — rigorous re-test of the headline claims

*Methods: ADF+KPSS stationarity; Toda-Yamamoto (1995) lag-augmented Granger causality with HAC; prewhitened cross-correlation; segmented ITS with HAC + placebo-cutoff null; user-cluster-robust OLS; per-year replication; permutation; split-half reliability; convergent validity vs pre-baked emotions + a 2nd independent model; BH-FDR. A claim is only kept if it survives these.*

## VERDICT SCORECARD (bottom line)

| Claim | Verdict | Strongest evidence |
|---|---|---|
| **English dreams darker than Russian** | 🟡 **MOSTLY MEASUREMENT ARTIFACT** | Back-translation of 500 dreams: **~84% of the gap is cross-lingual scorer calibration** on emotional content (neutral parallel sentences calibrate perfectly, offset −0.001; real dreams show a +0.10 language effect). Only ~16% (+0.02) is a residual content difference. The robust, novel result is now a **methodological caution** (see `docs/VALIDATION.md` §crown, `language_validation.md`). |
| **Multilingual sentiment scorers miscalibrate on affect** | ✅ **NEW METHOD FINDING** | Scorer is invariant on neutral parallel sentences (offset −0.001) but shows a systematic +0.10 offset on emotional dream content → a cautionary result for ALL cross-lingual dream/sentiment research |
| **Friday lighter than Monday** | ✅ **SUPPORTED (English, uncorrected)** | **Re-estimated 2026-08-21** with inference matched to the design (weekday is a property of the date, not the person): EN β=+0.0242, date-clustered p=.0083, date-block permutation p=.0080; same sign every year. Pooling *attenuates* (+0.0212); **RU is null (p=.54)** so this is not cross-lingual. Day-level β=+0.0187 (p=.068) → the null 7-day periodogram is a power ceiling. Old figures (β=+0.021, p=.009, perm .006) used user-clustered SEs + a report-level permutation of an unadjusted mean difference |
| **85% of dreams are negative** | ✅ **ROBUST** | holds across all 5 corpora (78–90%) |
| **Sentiment instrument is valid** | ✅ **at weekly+** | convergent (pre-baked negativity −0.57, joy +0.45; 2nd model r=0.53); reliable monthly, **noisy daily** |
| **Dreams darkened over decades** | ⚠️ **CONFOUNDED** | raw −0.004/decade (p=.10); within-collection trend present but era-bound → directional only |
| **Collective dreamed the pandemic** | ⚠️ **NOT ROBUST** | shift +0.017 (HAC p=.048) but placebo-in-time p=.63 with our generic measure |
| **Dreams couple with / predict markets** | 🔬 **FRONTIER (not yet established)** | Toda-Yamamoto null in EN/all; RU hits are numeric artifacts; the current daily/weekly signal is too noisy (§reliability). Kept as a frontier — revisit at monthly resolution once volume grows |
| **Dream time-series reliability is METRIC-DEPENDENT** | ⚠️ **use the right metric** | Hardened split-half (mean of 200 splits; `reliability_audit.csv`), level (detrended-deviation) D/W/M: **threat 0.23/0.62(0.27)/0.90(0.69), weapon 0.28/0.66(0.41)/0.92(0.77), fear 0.21/0.57(0.32)/0.88(0.71)** vs **sentiment 0.11/0.26/0.52, negativity 0.11/0.21/0.53**. ⇒ the old "daily r=0.09 → monthly+ for everything" was over-general: content-threat metrics are reliable weekly (deviation-reliability lower, so prefer monthly where possible); sentiment/negativity stay noisy |
| **Dream biometric — dream reports re-identify their author** | ✅ **CONFIRMED** (bold Q2) | **Replicated in 3 independent corpora under a temporal split**: DreamSeer 56× (329 users), DreamBank journals 40× (133 individuals; aggregates excluded), Reddit r/Dreams 14× (reposts dropped) — all perm p=.001. Content-vs-style ablation: content-only 66× > style-only 48× → not mere idiolect. Scope: identifiability of dream *reports* (privacy result) |
| **Prehistoric nightmare (Revonsuo)** | 🔬 **exploratory** (bold Q10) | dream threat content decoupled from real lethality (Spearman +0.04, ns); ancestral ~2.9× modern (2.7× breadth-normalized); disease 40M deaths→0.4% of dreams; EN↔DreamBank profile rho=.84. Mortality mapping order-of-magnitude; null n≈18 |
| **Age from dreams** | 🔬 **Strong (coarse)** (bold Q15) | child-vs-adult ROC-AUC **0.84** (person-grouped CV, p=.005); fine-grained age weak (MAE 14.7 vs 16.7y baseline; 5-band bal.acc 0.35). DreamBank age series-level + cohort-confounded |
| **"Universal nightmare" (Jung)** | 🔬 **reframed → measurement consistency** (bold Q12) | threat-motif ranks agree across EN corpora/eras (rho=0.90) **but neutral words agree as much (rho=0.92)** → shared-lexicon stability, NOT special nightmare universality; cross-lingual EN↔RU rho=0.16 open |
| **Collective synchrony — "do we dream together?"** | 🔬 **MOSTLY NULL — now BOUNDED, with a dissenting tail** (bold Q1) | weekday-matched null + block-bootstrap: EN 601 nights excess +0.0007 (95% CI −0.0024..+0.0037), Stouffer p=.28 → no convergence *on the mean*. **Power (new): the EN CI upper bound +0.0037 is 3.7× below the RU point estimate +0.0136** → bounded null, not merely underpowered. **Two 2026-08-21 corrections:** (1) the "RU peak nights track war news (2025-02-24)" attribution is **WITHDRAWN** — that date is an *EN* peak night (n=27, z=2.88) and is not scored at all in RU; no RU by-night table had ever been emitted. Computed RU peaks: 2025-11-16/13/28, 06-17, 08-27… clustered but unexplained. (2) the 44/601-vs-30.1 nominal excess was untested and **is significant** (binomial p=.0084; block-bootstrap rate .073 [.052,.097]) → mean null but **tail enriched 1.46×**; lower tail 1.16× and SD(z)=1.10 imply partial over-dispersion, asymmetry unexplained. date-shuffle control ≈0 |
| **Full-moon / geomagnetic-storm dream myths** | ✅ **NULL — debunked** (bold Q3) | 620 days; **nothing survives FDR**; continuous illumination flat; the 2 nominal full-moon hits are wrong-signed (lighter), fail FDR, absent in dose-response, and mirrored by the new-moon control; Kp/storm null (placebo p .09–.95) |
| **Day-residue — news themes → dream themes** | 🔬 **Frontier — weak hint, underpowered** (bold Q4) | **Re-examined (2026-07-19 audit):** original over-conservative in metric+FDR, but the fair test (reliable composite + adoption-detrend symmetric with coupling + focused family) leaves the **pre-specified family NULL** (0/15; best conflict→threat r=+0.38, 95% CI −0.00..+0.66, p=.074). Exploratory weekly conflict→weapon/fear are positive & correctly-lagged (r≈0.43–0.46, CIs exclude 0) but at the circular-shift p-floor (1/n≈.037), winner's-curse-selected — suggestive, not significant. Survives adoption-detrend (unlike coupling). Extend GDELT pre-2026 |
| **Physics of dreams (Zipf, low-D manifold)** | 🔬 **exploratory** (bold Q5) | Zipf exponent EN 1.25 / RU 1.09 (R²>.99); intrinsic dim ≈25 (95% CI ≈25–27; ambient 384; shuffled null 112). **Long-memory NULL:** Hurst H≈0.6 fails a short-memory AR(1) surrogate (p_ar1≥.23) — the earlier shuffle-only p=.016 doesn't survive |
| **Weekly rhythm across the affect family** | 🟡 **EN only — "cross-lingual" WITHDRAWN** (bold Q6) | neg_sentiment Fri−Mon −0.025 (joint p=.029; the −0.025 vs −0.024 gap vs the contrast model is an *estimator* difference, not a sample filter). The EN↔RU sign-agreement is **no longer counted**: re-estimation gives RU β=+0.0089, p=.54 — agreement between a significant estimate and a null one. Family-wide test does NOT clear FDR (best q≈.12), but that family is of *joint 6-df weekday Wald tests*, so it never corrected the Fri−Mon contrast, which is reported uncorrected. Spectral 7-day peak weak (p=.08–.80) — reconciled as a power ceiling by the new day-level arm |
| **Dreams↔markets weekly coupling (redo)** | 🔬 **Frontier / NULL** | 6 level hits at q<.10 ALL die under first-differencing (Δ q=.98); TY null; VIX/S&P/EPU differenced null → spurious co-trend (reaffirms the r=−0.44 artifact) |
| **Collective synchrony — within week** | 🔬 **NULL** (bold Q7) | EN/RU/pooled all null (excess ≈0, Stouffer p≈.56); RU within-DAY positive does NOT persist weekly. The **"day-specific news-residue" gloss is withdrawn** (see Q1 — the date it rested on belonged to the EN cohort); the non-persistence is reported without a mechanism. "We dream alone" holds at both scales *on the mean* |
| **Within-person bilingual darkness (EN vs RU)** | ✅ **corroborates F0011** (bold Q8) | 28 bilingual users: naive RU−EN gap +0.112 (Wilcoxon p=.001) → after F0011 calibration, content residual **+0.009 (95% CI −0.045..+0.059 → indistinguishable from 0 at n=28; ~92% scorer)** → darker-in-English is mostly measurement even within one brain (corroborates, doesn't prove zero). NEW caution: one-way MT-then-score LIGHTENS affect (+0.084) → invalid; use native+calibration |

**Headlines you can defend against any statistician (post-back-translation):**
- *"Across 154,069 dreams from five independent sources, ~85% carry negative emotion."* (robust)
- *"Collective dream mood is reliably most negative on Monday, lightest on Friday."* (p=.009, permutation, every year)
- *"Multilingual AI sentiment models are calibrated on neutral text but systematically miscalibrated on
  emotional content — so apparent cross-language dream differences are mostly the measuring instrument."*
  (neutral offset −0.001; emotional-content language effect +0.10; ~84% of the EN-RU gap). **A novel,
  field-relevant caution — and a reason to trust our WITHIN-language results.**

> **The EN-vs-RU "you dream darker in English" claim is NOT deleted — it is re-tiered** to
> "mostly measurement artifact, small residual." That is the honest, requested outcome of validating
> on neutral sentences.

Claims are **tiered, not deleted** (novelty/resonance choose the order; robustness chooses the label):
✅ Confirmed · ✅ Strong · 🔬 Frontier. The market-coupling and pandemic claims are held as **Frontier**
(intriguing, not yet established) pending more signal — see `docs/NOVELTY.md` for the roadmap.

## Claim: dreams couple with / predict societal signals (VIX, EPU, news tone, yields)
- **Verdict: NOT ESTABLISHED as lead-lag CAUSALITY — 6 TY tests pass FDR, but ALL are in the small/noisy RU cohort and mostly numeric artifacts (differenced near-deterministic macro series, p~1e-25); ZERO credible causal couplings in the primary EN/all cohorts. VIX↔dreams TY-null both directions; prewhitened CCFs collapse to noise.**
- Toda-Yamamoto tests significant after FDR: 0. Table: `60-results/tables/validation_coupling.csv`.
- Example: VIX↔dream valence TY p≈0.57 both directions; prewhitened CCF max|r|≈0.24 (noise).
- **⟳ REFRAMED then SHARPENED (2026-07-19, F0037) — "spurious" ≠ "fake", but here it's generic co-drift.**
  Per Alex's socionomic steer (dreams & markets as co-indicators of one *unobserved* mood), re-tested as a
  **latent common factor in LEVELS** with **autocorrelation-preserving** surrogates (phase-randomized;
  GETAB NoiseSim/SCA) **+ Engle-Granger cointegration + a NEUTRAL-theme negative control**. A **real shared
  low-frequency trend exists** (threat/negativity cointegrate with the market-mood PC1, p=.015/.005) — so
  "artifact/fake" was too strong. **BUT it is NOT anxiety-specific:** neutral dream themes (food p=.002,
  flying p=.0003) cointegrate as strongly, and the anxiety magnitude is at/below colored-noise chance once
  adoption-detrended. ⇒ most parsimoniously **generic composition/adoption co-drift** of a growing app vs
  trending 2024–26 markets, NOT an established dream-mood coupling. Correct status: **underdetermined /
  non-specific at this window**; distinguishing a true common mood from co-drift needs a longer window +
  content-specificity + cross-cultural (geotone pre-2026) distance.

## Claim: the collective dreamed the pandemic (COVID onset)
- **Verdict: NOT ROBUST (suggestive only) — anxious-content level shift at WHO date = +0.017 (HAC p=0.048); placebo-in-time (pre-COVID cutoffs) empirical p=0.625 over 32 fake dates; a shift this size is common at random pre-event dates, so our GENERIC measure can't attribute it to COVID. (Mallett's specific dysphoric-flair coding may be stronger.)**
- Segmented ITS anxious-content level shift at 2020-03-11: +0.017 (HAC p=0.048); placebo-cutoff empirical p=0.625 over 32 fake dates.

## Claim: weekly rhythm — Friday lighter than Monday
- Friday vs Monday (user-cluster-robust OLS, +season +textlen): β=+0.021 (p=0.0086); permutation p=0.0055.
- Per-year Fri−Mon gap (should be same sign): {2024:+0.040, 2025:+0.017, 2026:+0.016}.
- **RE-ESTIMATED 2026-08-21** (`analyses/2026-08-21-07-weekday-rhythm.py`) — the two figures above use
  inference that does not match the design. Weekday is a deterministic function of the *date*, so
  clustering on user leaves date-level dependence in the residuals, and the permutation shuffled
  weekday labels across *reports* (destroying date clustering rather than preserving it under the null)
  while permuting an unadjusted mean difference rather than the adjusted β it was quoted beside.
  Corrected:
  - **EN (headline, per the language-stratification rule): β=+0.0242**; p=.0067 user-clustered,
    **.0083 date-clustered**, .0069 two-way; **date-block permutation p=.0080**; 6,819 reports over
    246 Fri/Mon dates. Date clustering multiplies the SE by only ~1.1 — the contrast is *between*
    dates, so within-date dependence has little to inflate.
  - Pooled: β=+0.0212, date-clustered p=.0113, perm p=.0145 → **pooling attenuates**; the old pooled
    number was conservative, not inflated.
  - **RU: β=+0.0089, p=.54** (131 days) → no evidence. The EN↔RU "sign agreement" previously cited as
    corroboration is agreement between a significant estimate and a null one; **withdrawn as support**.
  - Day-level (daily means, ≥20 reports/day, EN): β=+0.0187, p=.068 on 163 days. The point estimate
    agrees with the report-level one; the precision does not — which explains the null 7-day
    periodogram peak (p=.077) as a power ceiling rather than as evidence against the contrast.
  - Language composition by weekday: RU share .190–.216 (χ² p=.019), but the Fri−Mon difference in
    that share is 0.25 pp — too small to move β given the RU−EN gap of +0.118.
- **Verdict: SUPPORTED (uncorrected, English) — survives date-clustered SEs, a date-block permutation
  of the adjusted coefficient, seasonal & length controls, and replicates across years. Not corrected
  for multiplicity, and not cross-lingual.** Note the 13-outcome q≈.12 family is a family of *joint
  6-df weekday tests*, not of Fri−Mon contrasts, so it never corrected this quantity.
- **Scope caution:** the working week most plausibly reaches a dreamer through sleep timing and sleep
  debt — a proximate bodily channel. It does not license the inference that this instrument can resolve
  attention-mediated societal drivers (news, markets); see the COVID placebo failure above.

## Claim: English dreams are darker than Russian dreams
- RU−EN (user-cluster-robust, +season +textlen): β=+0.118 (p=1.5e-49); positive => RU less negative than EN.
- Per-year RU−EN gap: {2024:+0.110, 2025:+0.117, 2026:+0.114}.
- **Verdict: SUPPORTED — robust to user-clustering, length & season, and replicates every year.**

## Claim: dreams have darkened over the decades (SDDb)
- Raw trend: -0.004/decade (p=0.095).
- WITH survey fixed effects (within-collection): -0.017/decade (p=2.3e-06).
- **Verdict: CONFOUNDED — the darkening largely reflects WHICH collections dominate each era; within-collection the trend persists. Directional only.**

## Measurement validity of the sentiment instrument
- Convergent vs DreamSeer pre-baked emotions: {'fear': -0.276, 'joy': 0.449, 'sadness': -0.201, 'anger': -0.135, 'negativity': -0.571, 'nightmare_index': -0.536} (expect −fear/−sadness/−negativity, +joy).
- Convergent vs a SECOND independent model (distilbert-multilingual): r=0.532.
- Split-half reliability: daily r=0.093, **weekly r=0.208** (n=124), monthly r=0.35 (n=29).
- **Verdict: MIXED — the instrument is convergent (vs pre-baked emotions & a 2nd model) and reliable at WEEKLY/monthly aggregation; the DAILY signal is mostly sampling noise (use weekly+). This is why daily coupling was null and why all time-series work must be weekly.**
