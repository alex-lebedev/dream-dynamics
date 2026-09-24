# ADR-0004 — A curated public showcase repository, separate from this one

- **Status:** proposed (blocked on the ETHICS §5 checklist — see Consequences)
- **Date:** 2026-08-24
- **Deciders:** Alex (owner); drafted with the research-engineer agent
- **Amends:** `docs/ETHICS.md` §2 rule 5 ("Private remote only")

## Context

The manuscript is approaching preprint. The owner wants a public GitHub presence carrying the
analysis scripts, the figures and videos, and a popular account of the findings, so that the work is
readable and partly reproducible by people outside the project.

`docs/ETHICS.md` §2 rule 5 currently states, without qualification, that if this repository is pushed
the remote is private. That rule was written when the only question was whether raw dream text could
escape, and it answers that question correctly. It does not distinguish between publishing the
repository and publishing a curated subset of it, so as written it forbids the intended release. A
rule that forbids the thing we are about to do needs amending on the record rather than being quietly
worked around, because the whole value of the rule is that it is not negotiable in the moment.

Three facts about the current state bear on the decision. This repository has **no remote configured
and has never been pushed**; a read-only audit of all 49 commits found no dream text, user
identifier, birthdate or sex field in the index or in reachable history, and
`scripts/check_release_cells.py --scope all` passes across worktree, index, HEAD and history. So the
tracked content is releasable. But **276 files under `60-results/` are untracked**, several of them
per-event tables with cell counts below the disclosure floor, and the entire `70-papers/` tree —
including the manuscript — has never been committed. A decision to make *this* repository public
would therefore be a decision about material that has not yet been reviewed under the release gate.

## Decisions

1. **Publish a separate repository, not this one.** The public artifact is a new repository built by
   an explicit export step, with its own history beginning at the export. This repository stays
   private and remote-less. Rationale: a curated export is reviewable as a whole before it ships,
   whereas flipping this repository public makes every past commit and every future `git add` a
   disclosure decision. It also means the private repository keeps its working mess — inbox CSVs,
   reviewer transcripts, N-of-1 artifacts — without that mess constraining what can be shown.

2. **`ETHICS.md` §2 rule 5 is amended, not deleted**, to read in substance: *this repository has no
   public remote; de-identified aggregates, code, figures and prose may be published only through the
   export step defined in ADR-0004, and only after the §5 checklist is closed.* The prohibition on
   raw dream text leaving the machine (§2 rule 1) is untouched and remains absolute.

3. **The export is a script, not a copy.** `scripts/build_public_release.py` assembles the public
   tree from an explicit allowlist and refuses to run if the release checker fails. Nothing reaches
   the public repository by being in a directory that happened to get copied. An allowlist fails
   closed; a denylist fails open, and the asymmetry matters more than the inconvenience.

4. **What ships:** the importable package and analysis scripts; de-identified aggregates already
   passing the min-N floor (`60-results/tables/*_public.csv`); figures and videos; the manuscript,
   the popular article and the docs; the data-cards. **What does not ship:** `10-data/raw/`,
   `10-data/external/`, `10-data/interim/`, every `*.parquet`, `00-inbox/`, `90-archive/`, the
   findings ledger and wiki log (they narrate unreleased intermediate results), and any table whose
   cells fall below the floor.

5. **Reproducibility is stated honestly rather than implied.** The public README says plainly which
   results a stranger can rebuild from what ships (anything computed from the public aggregates) and
   which they cannot (anything requiring DreamSeer raw or licensed third-party corpora), and points
   at the original sources rather than re-hosting them.

## Consequences

- **This ADR does not authorise publication.** It defines *how* to publish. The ETHICS §5 checklist —
  IRB determination, confirmation that DreamSeer's terms of service and consent cover secondary
  research use, third-party licence review, and the data-availability statement — is still entirely
  unchecked, and every item is a determination the owner must make. Publication remains blocked until
  it is closed; the checklist is the gate and this ADR is the mechanism.
- A licence must be chosen. `pyproject.toml` currently declares "Proprietary — private research
  repository", which contradicts a public release and must be updated together with a `LICENSE` file.
  Code and prose may want different terms (a permissive or copyleft software licence; a Creative
  Commons licence for the text and figures).
- Third-party licences constrain the export independently of our own choices. Several sources permit
  aggregates but prohibit redistributing raw text (DreamBank, Mallett's Reddit corpus, the
  r/confession baseline); Wikipedia-derived material carries share-alike; two API-sourced series
  (SerpApi, TMDB) have terms that may restrict redistribution of fetched series at all. The export
  ships no third-party raw text, which resolves most of this, but the data-availability statement
  must name each source and its terms.
- The public repository will drift from the private one. That is acceptable and expected: it is a
  release artifact, rebuilt by re-running the export, not a branch to be merged.
- The 276 untracked files under `60-results/` remain a standing hazard for as long as they are
  untracked. They are outside the allowlist, so they cannot reach the public repository through the
  export — but they are still one `git add -A` from the private history, and the release checker
  warns rather than fails on some of them.
