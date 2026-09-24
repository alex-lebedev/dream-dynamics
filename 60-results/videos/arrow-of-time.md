# Video 1 — the emotional arrow of time

DreamSeer English, 3,000 reports, true per-sentence XLM-R (`cardiffnlp/twitter-xlm-roberta-base-sentiment`) resampled to 20 points of normalized dream time.

Sample: the language-stratified arc cache (`sentence_sentiment.load_or_build_sentence_sentiment`), which draws up to 3,000 documents per language and admits reports of 5–25 sentences. This is the same instrument as the manuscript's primary arrow analysis but a different sample: the headline estimates in Section 3.3 use the first 6,000 qualifying documents per corpus in file order with a four-sentence minimum. The video therefore animates the regularity, not the reported effect size — for that, cite the manuscript.

- Mean end-minus-start drift, real sentence order: **-0.1157** (SE 0.0078)
- Mean end-minus-start drift, order shuffled within report: **-0.0079** (SE 0.0075)

The shuffled arm preserves each report's sentence scores and length and permutes only their order, so the gap between these two numbers is attributable to narrative sequence rather than to the sentiment instrument or the marginal distribution of scores.

The running mean shown on screen starts at n = 20, not at a single report: under `docs/ETHICS.md` §4 no displayed aggregate may rest on fewer than 20 reports or 5 distinct contributors, and a mean over a handful of reports approaches publishing those trajectories individually. The sample spans 974 contributors.

Report-level inference in the manuscript is author-clustered (Section 3.3); the standard errors above treat reports as independent and are shown only to indicate the precision of the animated running mean.
