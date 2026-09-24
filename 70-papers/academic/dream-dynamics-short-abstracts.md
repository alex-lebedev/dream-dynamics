# Short abstract and significance statement

The paper is a preprint and its own Abstract (~600 words) and Significance statement (~390 words) are
deliberately long: every number in them is carried by Section 3, and nine adversarial review passes installed
qualifiers there that we did not want to cut on our own initiative. Short forms are still needed — for indexes,
talks, repository landing pages and press — so this file holds them at fixed lengths, cut once and deliberately,
and checkable against the long forms rather than improvised each time.

**What the cut costs, stated plainly.** The short forms drop: the per-corpus closing-step values; the
sentence-count-matched and uncapped sensitivity results; the pooled-BH figure; the design-effect bound on the
comparator clustering asymmetry; the self-normalization check; the intrinsic-dimension and percolation bands;
the demographic axis; and the coupling null. None of those are cut because they are inconvenient — each is
stated in full in the Results, the Limitations, or the tiering summary. But a reader who sees only the short
abstract will not know that two of the five dream samples carry nothing, that the flagship's breadth is three
corpora (two from one platform), or that two conjuncts of the hypothesis we started from were contradicted.

**Two things are *not* cut, at any length, because dropping them makes a short form false rather than
merely brief.** The ending-deletion result is stated as deepening the descent rather than as making it
uniformly steeper, because DreamBank's closing step is negative under the primary instrument; and the
machine's closing lift is stated as the only one large enough to reverse a corpus's mean direction rather than
as the only resolution, because Reddit r/Dreams closes significantly higher than the waking comparator. The
short forms below use those formulations, which cost three words each. **Wherever only the short form is used,
the third and fourth sentences of the "Optional additional sentences" block below should be restored before
anything else.**

---

## Abstract — PNAS-scale (250-word limit; this draft is 277)

To land under 250 without losing a claim: drop the parenthetical AUC range (−5), compress "raw and
standardized by each corpus's own sentence-valence dispersion" to "raw and dispersion-standardized" (−7),
drop "in all five corpus-language samples" from the first sentence, which the "three of the five" sentence
already implies (−6), and cut "Dream corpora have rarely been characterized with the estimators of
complex-systems science" (−13). Do not recover words from the two protected formulations above.


> Dream corpora have rarely been characterized with the estimators of complex-systems science. We analyzed
> 75,241 reports — 30,241 from a multilingual journaling platform (5,524 users; English and Russian) and
> 45,000 from DreamBank, Reddit r/Dreams and the Sleep and Dream Database — under one instrument, against
> shuffles, surrogates and three matched non-dream corpora run through the identical pipeline.
>
> Dream reports end darker than they begin, in all five corpus-language samples (forward-versus-reversed
> AUC 0.647–0.667). The effect survives clustering on the contributor, re-scoring every sentence with the
> primary sentiment model, and seven arms that delete the report's ending, which deepen the descent rather
> than removing it (52 of 53 cells). The comparators both scale the effect and take part of it away: waking first-person
> accounts, fiction and encyclopedic openings all drift negative, so a weak descent is generic to prose.
> Its depth is not. Three of the five dream samples descend two to five times further than
> every comparator on the endpoint statistic, raw and standardized by each corpus's own sentence-valence
> dispersion (9 of 9 contrasts, *q* ≤ .018); the other two separate from the waking comparator on no descent
> estimand. We then tested and rejected our own structural reading — that dream reports fail to resolve at
> the close. Measured directly, closing steps separate no dream corpus from waking narrative — and are small and
> mostly positive; the only one large enough to reverse a corpus's mean direction is the machine's, whose
> automated interpretations lift in their final sentence alone by three times the largest human value. Two geometric findings — low intrinsic dimension and a
> percolating semantic continent — dissolve against matched natural language and are reported as calibration.

### Optional additional sentences, in restoration priority order

Use these to fill a longer limit (300–400 words), highest priority first. Sentences 3 and 4 are the ones whose
absence most changes what a reader would conclude.

1. *(scope of the geometry demotion)* Length-matched, every human corpus we measured occupies between 20.9 and
   27.3 intrinsic dimensions of a 384-dimensional embedding, and every human narrative corpus percolates;
   only the non-narrative comparator fragments.
2. *(the estimand that disagrees)* The whole-trajectory slope separates in only 6 of the same 9 contrasts, so
   the magnitude claim is an endpoint claim and we report both.
3. *(breadth bound — restore first)* The three corpora carrying the magnitude claim are two Dreamseer
   languages and eighteen archival diarists, which bounds its breadth.
4. *(the contradicted prediction — restore second)* We expected matched waking prose not to darken; every
   comparator does, so the flagship survives in a magnitude form rather than the categorical one we started
   from.
5. *(the collective frontier result)* Aggregate nightmare variance rises through the run-up to twelve of the
   corpus's own mood-darkening episodes (Kendall τ = +0.236) — a property of the population, held by no
   report in it — but it lacks the autocorrelation signature that would license a critical-slowing-down
   reading, and its significance is null-dependent: *q* = .024 against surrogates preserving linear serial
   dependence, *q* = .160 against a block bootstrap that also carries the series' own volatility. If this
   sentence is used at all it must be used whole; the first clause alone overstates it.
6. *(the coupling null)* Against a panel of market, news, search and cultural indicators, the aggregate
   weekly affect and content signal showed no coupling detectable at the effect sizes the design could
   resolve; the arrow and geometry layers were not tested against external indicators at all.

---

## Significance statement — 120-word target (PNAS limit; this draft is 137)

To reach 120: drop "and then try hard to destroy the one regularity that stands out" (−13) and "Matched
waking accounts, fiction and encyclopedia openings all drift downward too, so the arrow is not dreaming's
alone" (−20), then restore the second as the first thing if a longer limit allows — without it the depth
claim reads as a claim about dreams rather than a claim about dreams *relative to matched prose*, which is
the only form the data support.

> Dream reports are usually read as isolated narratives or as clinical material. We treat 75,000 of them as a
> population, and then try hard to destroy the one regularity that stands out. Dream reports end darker than
> they begin — not because they stop at the moment of waking, since deleting the ending deepens the descent
> rather than removing it, and not because of word choice, since shuffling each report's sentences abolishes it. Matched
> waking accounts, fiction and encyclopedia openings all drift downward too, so the arrow is not dreaming's
> alone. Its depth is: three of five dream samples fall two to five times further than any comparator. The
> more interesting story we wanted — that dreams fail to *resolve* — we measured, and refuted. Only the
> machine's closing lift is large enough to reverse a corpus's mean arc.

## Significance statement — 200-word variant (this draft is 211)

> Dream reports are usually read as isolated narratives or as clinical material. Here we treat more than
> 75,000 of them, from a multilingual journaling platform and three independent sources, as a population —
> and then try hard to destroy the one regularity that stands out. Dream reports end darker than they begin:
> in five of five corpus-language samples under a screening instrument, and four of five when every sentence
> is re-scored with the primary one. The descent is not an artifact of reports ending at the moment of waking,
> since deleting the ending deepens it in 52 of 53 cells rather than removing it, and not an artifact of word choice, since shuffling each
> report's sentences abolishes it. Nor is it simply what prose does, though prose does some of it: matched
> waking accounts of real events, literary fiction and encyclopedic openings all drift weakly downward. What
> is not generic is the depth. Measured in units of each corpus's own ordinary sentence-to-sentence movement,
> three of five dream samples fall two to five times further than every comparator. We also report the more
> interesting story we wanted and could not have: measuring the closing sentence directly refutes it. The only closing
> lift large enough to reverse a corpus's mean direction is the machine's.

---

## Title, at three lengths

- **Full (as written):** Measuring a dreaming population: a replicated emotional arrow of time in 75,000
  dream reports
- **Medium:** Measuring a dreaming population: an emotional arrow of time in 75,000 dream reports
- **Short:** An emotional arrow of time in dream reports

The title carries flagship-tier results only, and that rule is what settles what goes in it. "A dreaming
population" names the object — the contribution is to treat the corpus as a population rather than as a pile
of narratives — and "dream reports" names the unit in the same line, so the ambition and the limit arrive
together and §5's "Reports, not dreams" is not contradicted by the cover page. The count is in the title
because the population framing is only credible at that scale.

Two words in it were chosen against alternatives. **"Measuring"** replaced "the physics of", which two
independent reviewers identified as the single most attackable phrase in the paper — the thing a hostile
expert screenshots before reaching the pre-emption in §1. The system framing survives without it, and
"measuring" is additionally the more accurate verb: what this paper does is specify and calibrate an
instrument. **"Replicated"** is doing load-bearing work rather than decorating: the flagship's strength is
that it holds across four platforms, two languages and two valence instruments, and that is the property a
skeptical reader most needs to see before the word "dream".

Three things stay out of the title, each for a stated reason. The **geometry demotion** is Calibration tier,
and a title that lists a flagship and a calibration result at equal weight is the one place the paper's own
tiering is visibly not applied. The **variance precursor** is Frontier tier on 12 episodes with a revision
threshold of 30: do not restore it under any wording, including the hedged "early-warning-like" form that §3.5
and F0033 use in-text, because a title's qualifiers do not survive citation and the unhedged reading — that
dreams warn of external events — is null in this paper twice over (F0050 forward event study, F0052
anticipation). The **machine's closing lift** is a control corpus, and the only phrasing short enough for a
title ("the resolution only the machine supplies") reintroduces the exclusivity error that pass seven removed,
since Reddit r/Dreams also closes above the waking comparator. All three remain fully stated in the Abstract,
the Results and the tiering summary.

## Provenance

Every number in this file appears in the manuscript and is re-derived from the released result tables by
`scripts/audit_manuscript_numbers.py`. The 75,241 total is 30,241 Dreamseer plus 45,000 external; the two- to
five-fold multiplier is the standardized-endpoint point-estimate ratio range (2.1–5.2) across the nine
flagship contrasts, and is descriptive rather than inferential — the intervals are wide and right-skewed, and
the difference in valence units is what carries inference.
