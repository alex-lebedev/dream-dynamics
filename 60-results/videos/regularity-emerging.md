# Video 3 — a population regularity condensing out of individual noise

DreamSeer English, 3,000 reports of 5–25 sentences from the language-stratified arc cache, split by **contributor** into disjoint groups of 1,517 reports (487 contributors) and 1,483 reports (487 contributors). The running mean of group A is compared against the final mean arc of group B, so the convergence measured is out-of-sample and no contributor is on both sides.

This is an illustration of a point the manuscript already makes in Section 3.4 — that the arc's *shape* is aggregate while its *direction* is not — and not a separate result. It is not in the manuscript, the findings ledger or the reproducibility table.

The sweep starts at n = 20 rather than at a single report: under `docs/ETHICS.md` §4 no displayed aggregate may rest on fewer than 20 reports or 5 distinct contributors, and a mean arc over a handful of reports is close to publishing those trajectories individually. The full sample spans 974 contributors.

- Distance to the held-out contributors' arc falls from **0.081** at n = 20 to **0.0134** at n = 1,517 (real sentence order).
- Fitted log-log slope over the first half of the range: **-0.33**, against the −0.50 that *independent* averaging predicts. The shortfall is expected: these 3,000 reports come from 974 contributors, so successive reports are not independent draws and within-contributor similarity slows convergence.
- Shuffled order converges too, onto a flat arc: **0.0092**. Convergence is not the claim; the *shape* converged onto is.

Comparing a running mean to its own endpoint would force the curve to zero by construction. A held-out set of *different contributors* cannot be driven to zero this way, so the decay is evidence that the mean arc is a stable population quantity rather than an artifact of accumulating the same reports or of recognising the same people.

Instrument and sample: true per-sentence XLM-R (`cardiffnlp/twitter-xlm-roberta-base-sentiment`), the same scorer as the manuscript's primary arrow analysis, applied to the language-stratified arc cache rather than to the 6,000-document per-corpus prefix behind the Section 3.3 estimates. The video animates the regularity; the reported effect sizes belong to the manuscript.

Report-level inference in the manuscript is author-clustered (Section 3.3). This video treats reports as exchangeable, which is appropriate for illustrating convergence of the population mean but is not the manuscript's inferential model.
