# ADR-0005 — Narrow the showcase to the figure-producing scripts, and generate its landing page

- **Status:** accepted
- **Date:** 2026-08-25
- **Deciders:** Alex (owner); drafted with the research-engineer agent
- **Refines:** `docs/decisions/0004-public-showcase-release.md` (which established *that* there is a
  separate public repository and how it is built; this decides *what goes in it*)

## Context

ADR-0004 settled the mechanism: a separate public repository, assembled by
`scripts/build_public_release.py` under an allowlist, gated by `check_release_cells.py` before and
after assembly. Its allowlist was drawn generously — all 105 analysis scripts, all 22 tooling
scripts, every figure under `60-results/`, both video encodings, and the 15.6 MB `.docx`.

Building it that way exposed three problems.

**The generous script list advertised a reproducibility that does not exist.** Of the 105 analysis
scripts, exactly one runs without data that will never ship. The rest read raw dream reports or the
interim embedding and sentiment caches derived from them. A visitor who clones a repository
containing 105 dated analysis scripts and no data reasonably concludes either that something is
missing or that the code is decoration. Both readings are worse than shipping fewer scripts and
saying plainly what each one can do.

**There was nothing for a visitor to look at.** The popular account is Markdown with four `<video>`
tags. GitHub's Markdown renderer strips them, so the videos — the most legible evidence the project
has — were invisible to anyone who did not clone the repository and open a local file.

**Two files could carry a disclosure leak past the gate.** `check_release_cells.py` scanned
`.md/.json/.txt/.tsv/...` but not `.html`, and the post-flight in the release builder scanned an even
narrower set. A generated landing page quoting percentages and sample sizes was precisely the file
most likely to be read and the only one not being checked.

## Decision

1. **Ship the ten scripts that draw what the release contains, not all 105.** The set is enumerated
   in `ANALYSES` in the builder, and an invariant (`assert_assets_accounted_for`) fails the build if
   a shipped document references an asset that is not shipped, or if an asset ships with no shipped
   document pointing at it. The importable package `src/psychohistory/` still ships in full, so the
   analysis *logic* is present even where the CLI wrapper that drives it is not.

2. **Be explicit in the README about what runs.** Three tiers: runs as shipped (one diagram); runs
   from the shipped aggregate tables (the panels that read `60-results/showcase/*.csv`); does not run
   (everything else, shipped to be read rather than executed).

3. **Track the eight aggregate tables the figure scripts read.** They were untracked, so the
   tracked-only rule — tabular data ships only if git-tracked, because tracked data has already
   passed the gate in the index, HEAD and history scopes — would have refused them, and the release
   would have shipped figure code with none of its inputs. All eight pass the gate.

4. **Generate `index.html` from the popular article** (`scripts/build_showcase_site.py`), never hand
   maintain it. The Markdown stays the single source of the prose, so the page cannot drift from the
   critic-gated text, and the page is scanned by the disclosure gate like everything else. Relative
   links are resolved against the article's own directory rather than prefix-matched, and the build
   fails on any link that does not resolve inside the assembled tree.

5. **Ship a rendered `manuscript.html` instead of the `.docx`.** Same citeproc path, 360 KB against
   15.6 MB, previewable in a browser, and diffable. The `.docx` remains a local build product for
   journal submission.

6. **Scan `.html` in both the gate and the post-flight**, with a self-test case pinning it.
   Generated renderings inherit their source's prose exemption *and its finding count* — a rendering
   may reproduce what its source is permitted to say and nothing more, so a rendering bug that
   introduced a new single-contributor cell still fails the build.

7. **Drop the `.webm` duplicates** (second encodings of committed MP4s, ~8 MB) and the loose
   exploratory PNGs no shipped document references.

8. **Fresh git history in the public repository, single initial commit.** The private history
   contains artifacts purged for disclosure reasons; importing it would republish them.

## Consequences

The release is 234 files and 29 MB, against 222 files with a far larger footprint before. Every
shipped image has a shipped script that draws it and a shipped document that references it, and that
property is enforced rather than asserted.

The cost is that the public repository is no longer a superset of the private one's code, so a
reader cannot audit an analysis that is not in the ten. That is the intended trade: the honest claim
"here are the ten scripts behind these figures, and here is exactly what you can rerun" is worth
more than 105 scripts none of which run.

Publication remains blocked on the `docs/ETHICS.md` §5 checklist, unchanged by this ADR. The tree
exists; deciding it may be pushed is still a human determination.
