"""Assemble the curated public release tree defined by ADR-0004.

The public repository is built, never copied. Everything that ships is named by an explicit
allowlist below; anything not named is absent by default. That asymmetry is the whole design — an
allowlist fails closed when someone adds a new directory of intermediate results, a denylist fails
open — and it is why this script is the only sanctioned path from the private repository to a public
one (`docs/ETHICS.md` §2 rule 5).

Three gates, in order. The build aborts on any of them and leaves nothing behind:

1. **Pre-flight.** `scripts/check_release_cells.py --scope all` must pass on this repository.
2. **Assembly.** Every candidate is checked against the hard denylist regardless of allowlist match,
   and tabular files must additionally be *tracked in git* — meaning they have already passed the
   release gate in the index, HEAD and history scopes.
3. **Post-flight.** The assembled tree is re-scanned from scratch with the same disclosure-floor and
   tripwire logic, as if it were an untrusted directory someone handed us.

    python scripts/build_public_release.py --dry-run          # report what would ship
    python scripts/build_public_release.py --out ../psychohistory-public

Publication remains blocked on the `docs/ETHICS.md` §5 checklist, which this script cannot and does
not evaluate. It builds a tree; deciding that the tree may be published is a human determination.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_manuscript  # noqa: E402
import build_showcase_site as site  # noqa: E402
import check_release_cells as gate  # noqa: E402

# --- what ships -------------------------------------------------------------------------------
# Globs are relative to the repository root. Categories marked TRACKED_ONLY additionally require
# the file to be tracked in git, because tracked tabular data has already been through the
# disclosure-floor gate in three scopes.
CODE = [
    "pyproject.toml",
    # The whole importable package, deliberately. Per the repository conventions the scripts are thin
    # CLI wrappers and the package holds the logic, so shipping the package retains the ingest,
    # feature and statistics code even where the wrapper that drives it is not released.
    "src/psychohistory/**/*.py",
]
# Every analysis script the manuscript's reproducibility table (Supplementary S1) cites, plus the
# scripts that draw the shipped figures and videos, so that every script path in the paper
# resolves in the public repository. Most of them read raw reports or caches derived from them,
# which never ship; the README says which scripts run from the shipped aggregates and which are
# released to be read. `assert_assets_accounted_for` still enforces that no image ships without
# the script that drew it.
ANALYSES = [
    "analyses/2026-07-18-10-validation-ts.py",
    "analyses/2026-07-18-11-validation-confounds.py",
    "analyses/2026-07-18-15-bold-synchrony-biometric.py",
    "analyses/2026-07-18-19-physics-of-dreams.py",
    "analyses/2026-07-18-20-collective-cycles.py",
    "analyses/2026-07-18-21-coweek-synchrony.py",
    "analyses/2026-07-18-23-crosslingual.py",
    "analyses/2026-07-19-01-biometric-replication.py",
    "analyses/2026-07-19-13-nightmare-contagion.py",
    "analyses/2026-07-19-17-dream-narrative-physics.py",
    "analyses/2026-07-19-18-dream-emotional-arcs.py",
    "analyses/2026-07-19-19-collective-dynamics.py",
    "analyses/2026-07-19-24-collective-wave-poc.py",
    "analyses/2026-07-20-13-barometer-v5-search-culture.py",
    "analyses/2026-07-20-15-forward-event-study.py",
    "analyses/2026-07-21-01-demographic-collective-dynamics.py",
    "analyses/2026-07-21-04-physics-of-dreams.py",
    "analyses/2026-07-22-01-arrow-arcs-external.py",
    "analyses/2026-07-22-02-figures-publication.py",
    "analyses/2026-07-22-03-arc-basis-control.py",
    "analyses/2026-07-22-04-figures-supp.py",
    "analyses/2026-08-20-01-arrow-user-clustered.py",
    "analyses/2026-08-20-02-arrow-awakening-audit.py",
    "analyses/2026-08-20-03-corpus-overlap.py",
    "analyses/2026-08-20-04-geometry-language-baseline.py",
    "analyses/2026-08-20-05-arrow-true-xlmr.py",
    "analyses/2026-08-20-06-figures-revision.py",
    "analyses/2026-08-21-01-arrow-contrast.py",
    "analyses/2026-08-21-02-arrow-recovery.py",
    "analyses/2026-08-21-03-arrow-denominator.py",
    "analyses/2026-08-21-04-precursor-hardening.py",
    "analyses/2026-08-21-05-synchrony-ru-peaks.py",
    "analyses/2026-08-21-06-branching-null.py",
    "analyses/2026-08-21-07-weekday-rhythm.py",
    "analyses/2026-08-22-05-chain-figure.py",
    "analyses/2026-08-22-10-covid-content-correspondence.py",
    "analyses/2026-08-22-11-dreamseer-content-events.py",
    "analyses/2026-08-22-12-nonlinear-coupling.py",
    "analyses/2026-08-22-14-nonlinear-trend-robustness.py",
    "analyses/2026-08-22-18-design-cost-audit.py",
    "analyses/2026-08-22-20-network-gallery.py",
    "analyses/2026-08-24-01-panel-axis-reliability.py",
    "analyses/2026-08-24-02-video-arrow.py",
    "analyses/2026-08-24-03-video-percolation.py",
    "analyses/2026-08-24-04-video-emergence.py",
]
# Tooling that a reader has a reason to run or audit: the disclosure gate, the builder that made the
# release, and the two document builders. Personal-workstation tooling (Zotero sync, Obsidian plugin
# installation) and the data-download CLIs are omitted.
TOOLS = [
    "scripts/check_release_cells.py",
    "scripts/build_public_release.py",
    "scripts/build_showcase_site.py",
    "scripts/build_manuscript.py",
    "scripts/build_references_bib.py",
    "scripts/transcode_animations.py",
    "scripts/fetch_language_baselines.py",   # cited in S1: acquires the non-dream corpora
]
# Named individually rather than globbed. `BOLD-QUESTIONS`, `INSIGHTS-AND-PRODUCTS` and `NOVELTY`
# are internal ideation and commercial-positioning notes, not research documentation, and a public
# research release is a worse document for containing them.
DOCS = [
    "docs/ARCHITECTURE.md",
    "docs/DATA-CATALOG.md",
    "docs/ETHICS.md",
    "docs/METHODS.md",
    "docs/VALIDATION.md",
    "docs/decisions/*.md",
    "RUNBOOK.md",
]
CARDS = [
    "10-data/manifests/*.json",
    "10-data/processed/**/*_schema.json",
    "10-data/processed/**/README.md",
]
PAPERS = [
    "70-papers/academic/dream-dynamics.md",
    "70-papers/academic/dream-dynamics-short-abstracts.md",
    "70-papers/academic/references-seed.bib",
    "70-papers/academic/references-numbered-archive.md",
    "70-papers/academic/nature-brackets.csl",
    "70-papers/popular/dreams-arrow-of-time.md",
]
# Only assets a shipped document actually points at. The `.webm` duplicates are dropped (8 MB of
# second encodings of committed MP4s) and so is the 15.6 MB DOCX, which is a build product of the
# Markdown and is rebuilt by `scripts/build_manuscript.py`.
VISUALS = [
    "60-results/showcase/pub/fig0_chain.png",
    "60-results/showcase/pub/fig1_scaling_manifold.png",
    "60-results/showcase/pub/fig2_geometry.png",
    "60-results/showcase/pub/fig3_arcs.png",
    "60-results/showcase/pub/fig4_arrow_robustness.png",
    "60-results/showcase/pub/fig5_csd.png",
    "60-results/showcase/pub/figS2_event_study.png",
    "60-results/showcase/pub/*.pdf",
    "60-results/showcase/12_dream_fingerprint_replication.png",
    "60-results/showcase/45_arrow_arcs_external.png",
    "60-results/network-gallery/figures/fig[1-5]-*.png",
    "60-results/network-gallery/figures/fig[1-5]-*.pdf",
    "60-results/network-gallery/figures/network-breathing.mp4",
    "60-results/videos/arrow-of-time.mp4",
    "60-results/videos/semantic-continent.mp4",
    "60-results/videos/regularity-emerging.mp4",
    "60-results/videos/*.md",
]
# The aggregate tables the shipped figure scripts read, so the figures can actually be redrawn. The
# `10-data/processed/**/*.csv` glob is gone: nothing under it is tracked, so it only ever advertised
# data that was not going to ship.
TABLES_TRACKED_ONLY = [
    "60-results/tables/*.csv",
    "60-results/tables/*.json",
    "60-results/showcase/*.csv",
]
LICENCE = ["LICENSE", "LICENSE.md", "LICENSE.txt", "LICENSE-CONTENT.md"]

# --- what can never ship, whatever the allowlist says -----------------------------------------
# Checked as path substrings against every candidate. `50-wiki/` and `90-archive/` are excluded
# because they narrate unreleased intermediate results and superseded claims respectively;
# `00-inbox/` holds raw exports and reviewer transcripts.
DENY_DIRS = ("10-data/raw/", "10-data/external/", "10-data/interim/",
             "00-inbox/", "90-archive/", "50-wiki/", "60-results/overnight/",
             # The ephemeris layer is out of the primary pipeline by ADR-0001 §2 and reserved for a
             # late-stage pre-registered negative control. It reached the first build through the
             # `processed/**/*_schema.json` data-card glob, which is exactly how an allowlist of
             # patterns leaks: nobody named it, a pattern matched it. Three reasons it must not
             # ship. It documents no analysis in this release. Its vocabulary is zodiac ingresses
             # and retrograde stations, so a reader who screenshots the file list has a story about
             # this project that the work does not support. And its moon columns are known broken
             # for 2024+ (see that directory's README), so it would be shipping wrong numbers too.
             "10-data/processed/celestial/",
             ".venv", ".git/", "__pycache__")
DENY_SUFFIX = (".parquet", ".npz", ".env", ".pkl", ".joblib")
# Narrow, not clever: these name the actual N-of-1 and raw-export artifacts. A broader pattern such
# as "personal_" also catches the personal-*narrative* baseline data-card, which is a description of
# a public comparison corpus and perfectly releasable.
DENY_NAMES = ("_partial.csv", "users_data", "dreams_data", "dream_level",
              "personal_n1", "personal_monthly", "personal_symbol", "personal_on_collective",
              # The event cohorts arrived from a previous project with each day annotated by
              # ephemeris: these five schemas advertise `retrograde_body_count`,
              # `celestial_sign_ingress_count` and `moon_phase_angle_deg` columns on the
              # earthquake, terror, coup, conflict and aviation tables. ADR-0001 §2 and
              # `docs/METHODS.md` §3A are explicit that this enrichment is *not* reused for
              # dream outcomes — controls are re-derived — so shipping the schemas would document
              # a layer the work does not use, on data that does not ship, in the vocabulary most
              # likely to be misread. The cohort, core, control-pool and outcome-day schemas
              # describing what *is* used are unaffected.
              "_enriched_daily")


# --- prose that discusses the disclosure rule rather than breaking it -------------------------
# The post-flight tripwire keys on single-contributor language, which cannot distinguish a released
# N-of-1 cell from a sentence *about* N-of-1 cells. Two documents legitimately discuss the rule and
# would otherwise block every build. Exemptions are per-file, carry a written reason, and are
# printed on every run — never silent. Anything not listed here still fails closed.
PROSE_EXEMPT = {
    "70-papers/academic/dream-dynamics.md":
        "§6 narrates the three disclosure-check failures, including the N-of-1 monthly series that "
        "was purged. It describes violations that were fixed and publishes no cell.",
    "docs/decisions/0004-public-showcase-release.md":
        "This ADR lists 'N-of-1 artifacts' among the categories the export excludes.",
    "docs/decisions/0005-showcase-scope-and-generated-site.md":
        "Decision 6 describes the exemption-inheritance rule, and describing it requires the phrase "
        "'single-contributor cell'. It is a specification of the check, not a cell.",
}

# Generated renderings inherit their source's exemption rather than carrying one of their own, and
# only up to the source's finding count. Writing a second literal exemption for `manuscript.html`
# would mean a rendering bug that introduced a *new* single-contributor cell went unnoticed, because
# the file it appeared in was already blanket-exempt. Inheriting with a ceiling keeps the derivative
# honest: the rendering may reproduce what the source is allowed to say, and nothing further.
GENERATED_FROM = {"manuscript.html": "70-papers/academic/dream-dynamics.md"}


def denied(rel: str) -> str | None:
    if any(d in rel for d in DENY_DIRS):
        return "denylisted directory"
    if rel.endswith(DENY_SUFFIX):
        return "denylisted file type"
    if any(n in rel for n in DENY_NAMES):
        return "denylisted filename pattern"
    return None


def clear_tree(out: Path) -> bool:
    """Empty the output directory but keep `.git/`. Returns True if a repository was preserved.

    An earlier version called `shutil.rmtree(out)`, which destroyed the git repository along with
    the files — every rebuild silently orphaned the public repo, discarding its history and, after
    the first push, its remote. Rebuilding a generated tree must not be a destructive act on the
    thing that publishes it.
    """
    if not out.exists():
        return False
    keep = out / ".git"
    had = keep.is_dir()
    for p in out.iterdir():
        if had and p == keep:
            continue
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    return had


def tracked() -> set[str]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True)
    return set(out.stdout.split())


def collect(tracked_set: set[str]) -> tuple[list[str], list[tuple[str, str]]]:
    """Return (files to ship, [(file, why skipped)])."""
    ship: set[str] = set()
    skip: list[tuple[str, str]] = []
    plain = [(g, False) for g in CODE + ANALYSES + TOOLS + DOCS + CARDS + PAPERS + VISUALS + LICENCE]
    gated = [(g, True) for g in TABLES_TRACKED_ONLY]

    for pattern, needs_tracked in plain + gated:
        for p in sorted(ROOT.glob(pattern)):
            if not p.is_file():
                continue
            rel = p.relative_to(ROOT).as_posix()
            why = denied(rel)
            if why:
                skip.append((rel, why))
                continue
            if needs_tracked and rel not in tracked_set:
                skip.append((rel, "untracked, so it has not passed the release gate"))
                continue
            ship.add(rel)
    return sorted(ship), skip


ASSET_RE = re.compile(r"60-results/[A-Za-z0-9_./-]+\.(?:png|pdf|mp4|webm|gif)")


def assert_assets_accounted_for(ship: list[str]) -> None:
    """Every asset a shipped document points at must itself ship, and vice versa.

    Two failure modes this catches, both of which happened while assembling this release. A document
    referencing an image that was trimmed from the allowlist renders as a broken link on GitHub,
    which reads as carelessness about the evidence. An image shipping without a shipped document
    pointing at it is an orphan whose provenance a reader cannot check. Both are build errors.
    """
    shipped = set(ship)
    docs = [r for r in ship if r.endswith(".md")]
    wanted: dict[str, list[str]] = {}
    for rel in docs:
        for m in ASSET_RE.findall((ROOT / rel).read_text(errors="replace")):
            wanted.setdefault(m, []).append(rel)

    missing = {a: srcs for a, srcs in wanted.items() if a not in shipped and (ROOT / a).exists()}
    orphans = [r for r in ship
               if r.startswith("60-results/") and r.endswith((".png", ".mp4")) and r not in wanted]
    if missing:
        for a, srcs in sorted(missing.items()):
            print(f"   BROKEN LINK  {a}  referenced by {', '.join(srcs)}", file=sys.stderr)
        raise SystemExit(f"{len(missing)} referenced asset(s) are not in the allowlist.")
    if orphans:
        for o in orphans:
            print(f"   ORPHAN ASSET {o}  (no shipped document references it)", file=sys.stderr)
        raise SystemExit(f"{len(orphans)} asset(s) ship with nothing pointing at them.")
    print(f"      {len(wanted)} referenced assets all present; no orphans")


def preflight() -> None:
    print("[1/3] pre-flight: check_release_cells --scope all")
    r = subprocess.run([sys.executable, "scripts/check_release_cells.py", "--scope", "all"],
                       cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-4000:], file=sys.stderr)
        raise SystemExit("pre-flight FAILED: this repository does not currently pass the "
                         "disclosure-floor check, so nothing may be exported.")
    print("      pass")


def postflight(out: Path) -> None:
    """Re-scan the assembled tree as if it were untrusted input."""
    print("[3/3] post-flight: re-scanning the assembled tree")
    bad: list[str] = []
    exempted: list[str] = []
    for p in sorted(out.rglob("*")):
        # `.git/` is preserved across rebuilds but is not part of the release surface: its objects
        # are compressed binary, which the text scanner would report as unreadable findings.
        if ".git" in p.parts:
            continue
        if not p.is_file():
            continue
        rel = p.relative_to(out).as_posix()
        found: list[str] = []
        if p.suffix.lower() == ".csv":
            found = gate.check(f"public:{rel}", str(p), from_git=False, show=False)
        elif p.suffix.lower() in (".md", ".json", ".txt", ".bib", ".html"):
            try:
                found = gate._scan_text(f"public:{rel}", p.read_text(errors="replace"))
            except Exception as e:                                    # unreadable is a finding
                found = [f"public:{rel}: unreadable ({e})"]
        if not found:
            continue
        # CSV findings are never exempt: an exemption is for prose about the rule, not for data.
        if p.suffix.lower() == ".csv":
            bad += found
        elif rel in PROSE_EXEMPT:
            exempted.append(f"{rel} ({len(found)}): {PROSE_EXEMPT[rel]}")
        elif rel in GENERATED_FROM and GENERATED_FROM[rel] in PROSE_EXEMPT:
            src = GENERATED_FROM[rel]
            ceiling = len(gate._scan_text(src, (ROOT / src).read_text(errors="replace")))
            if len(found) <= ceiling:
                exempted.append(f"{rel} ({len(found)} <= {ceiling}): rendering of {src}, which is "
                                f"exempt — {PROSE_EXEMPT[src]}")
            else:
                print(f"   the rendering of {src} has {len(found)} findings but the source has "
                      f"{ceiling}; the extra one is not covered by the source's exemption",
                      file=sys.stderr)
                bad += found
        else:
            bad += found

    for e in exempted:
        print(f"      exempt  {e}")
    if bad:
        for b in bad[:40]:
            print("   VIOLATION", b, file=sys.stderr)
        clear_tree(out)
        raise SystemExit(f"post-flight FAILED with {len(bad)} finding(s); output tree emptied.")
    print("      pass")


PUBLIC_README = """# Measuring a dreaming population

Figures, videos, de-identified aggregates and the code that drew them, for *{title}*.

- **Start here:** [`index.html`](index.html) — the illustrated account, with the videos
  ([read it rendered]({pages}))
- **The manuscript:** [`manuscript.html`](manuscript.html) (citations resolved) or the source,
  [`dream-dynamics.md`](70-papers/academic/dream-dynamics.md)
- **Videos:** [`60-results/videos/`](60-results/videos/)

## What this is

A population of people wrote down their dreams in an app, daily, for two years — {n_app} reports
from {n_users} users. Set beside {n_ext} more from three public archives, that is the corpus. This
repository holds what could be measured in it without ever publishing one of those reports: the
emotional shape of a dream report over its own length, the geometry of the language it is written
in, and a set of tests for whether any of it tracks the news.

Some of those tests came back positive and some came back null. The null results are in the paper
and in the article, in the same detail as the positive ones.

## What is here, and what is not

This is a **curated showcase**, not a mirror of the working repository. It ships the importable
package (`src/psychohistory/`), the {n_scripts} analysis scripts cited in the manuscript's reproducibility table (including those that draw every figure and video it ships),
the aggregate tables those scripts read, the data-cards describing each source, the methods and
ethics documents, and the manuscript.

It does **not** contain dream reports, and it never will. The primary corpus is personal narrative
written by identifiable people; raw text and user identifiers do not leave the machine they were
analysed on. No cell published anywhere in this repository rests on fewer than
{min_reports} reports or {min_users} contributors — see [`docs/ETHICS.md`](docs/ETHICS.md), and
[`scripts/check_release_cells.py`](scripts/check_release_cells.py), which is the check that enforces
it and which refuses to build this release if it fails.

Also absent: the other {n_other} analysis scripts in the working repository (exploratory passes
and superseded drafts), the corpus-download and feature-build tools, and the working notes.

## What you can and cannot rebuild

Being exact about this, because "analysis code" in a repository with no data usually means less than
it sounds like:

| | |
|---|---|
| **Runs as shipped** | `2026-08-22-05-chain-figure.py` — the measurement-chain diagram draws from nothing but itself. |
| **Runs from shipped tables** | The panels of `fig1`, `fig2_geometry` and `fig4_arrow_robustness` that read the aggregates in `60-results/showcase/*.csv`. Those CSVs are the numbers behind the claims; you can check the figures against them, and check them against the paper. |
| **Does not run** | Everything else. The corpus figures need the raw reports; the videos need cached sentence-level embeddings and sentiment (tens of GB, derived from raw text). They are shipped to be *read* — so that what was computed is inspectable — not to be executed. |

For the external corpora, go to the sources rather than to us: DreamBank, the Sleep and Dream
Database, and the published Reddit dream corpora. Each is described with its licence and retrieval
terms in [`10-data/manifests/`](10-data/manifests/).

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[signals,external,dev]"
```

## Licence and citation

Code is MIT ([`LICENSE`](LICENSE)); prose, figures and videos are CC BY 4.0
([`LICENSE-CONTENT.md`](LICENSE-CONTENT.md)). Cite the manuscript for the findings, and each data
source on the terms in its own data-card.

Built by `scripts/build_public_release.py` from a private research repository; the file list is in
[`RELEASE-MANIFEST.txt`](RELEASE-MANIFEST.txt).
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="../dreaming-population",
                    help="output directory (default: sibling of this repository)")
    ap.add_argument("--pages-url", default="https://alex-lebedev.github.io/dreaming-population/",
                    help="GitHub Pages URL to link from the README")
    ap.add_argument("--dry-run", action="store_true", help="report the file list and stop")
    a = ap.parse_args()

    tracked_set = tracked()
    ship, skip = collect(tracked_set)

    if a.dry_run:
        print(f"[dry-run] would ship {len(ship)} files; skipped {len(skip)}")
        by_top: dict[str, int] = {}
        for rel in ship:
            by_top[rel.split("/")[0]] = by_top.get(rel.split("/")[0], 0) + 1
        for k, v in sorted(by_top.items()):
            print(f"   {v:5d}  {k}")
        print("\n   skipped (first 25):")
        for rel, why in skip[:25]:
            print(f"      {rel}  —  {why}")
        return 0

    if not any((ROOT / n).exists() for n in LICENCE):
        raise SystemExit(
            "no LICENSE file in the repository. ADR-0004 requires an explicit licence choice "
            "before a public release is assembled; pyproject.toml currently declares "
            "'Proprietary'. Add a LICENSE (and update pyproject) or run with --dry-run."
        )

    assert_assets_accounted_for(ship)
    preflight()

    out = Path(a.out).expanduser().resolve()
    if out == ROOT or ROOT in out.parents:
        raise SystemExit(f"refusing to write inside the source repository: {out}")
    kept_repo = clear_tree(out)
    print(f"[2/3] assembling {len(ship)} files -> {out}"
          + ("  (existing .git preserved)" if kept_repo else ""))
    for rel in ship:
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dst)

    print("      generating the landing page and the rendered manuscript")
    site.build(out=out / "index.html")
    build_manuscript.render_html(out / "manuscript.html")
    broken = site.check_links((out / "index.html").read_text(), out)
    for b in broken:
        print(f"   BROKEN LINK  index.html -> {b}", file=sys.stderr)
    if broken:
        clear_tree(out)
        raise SystemExit(f"{len(broken)} link(s) on the landing page do not resolve in the "
                         f"assembled tree; output emptied.")

    title = "Measuring a dreaming population: multilevel temporal organization in 75,000 dream reports"
    n_all = len(list((ROOT / "analyses").glob("*.py")))
    (out / "README.md").write_text(PUBLIC_README.format(
        # Corpus composition as stated in the manuscript and the popular account: the app stream is
        # 30,241 of the ~75,000 reports, and saying "75,000 people wrote them in an app" would be
        # wrong about three of the four sources.
        title=title, pages=a.pages_url, n_app="30,241", n_users="5,524", n_ext="roughly 45,000",
        n_scripts=len(ANALYSES), n_other=n_all - len(ANALYSES),
        min_reports=gate.MIN_REPORTS if hasattr(gate, "MIN_REPORTS") else 20,
        min_users=gate.MIN_USERS if hasattr(gate, "MIN_USERS") else 5))
    # GitHub Pages runs Jekyll over a branch deployment unless told not to, which strips paths it
    # reads as templates and slows every build for a site that is one static file.
    (out / ".nojekyll").write_text("")
    (out / ".gitignore").write_text(
        "# This tree is generated by scripts/build_public_release.py in the private research\n"
        "# repository. Edit nothing here; edit the source and rebuild.\n"
        "__pycache__/\n*.py[cod]\n.venv/\n.DS_Store\n")
    manifest = "\n".join(ship)
    (out / "RELEASE-MANIFEST.txt").write_text(
        f"# {len(ship)} allowlisted files, built by scripts/build_public_release.py (ADR-0004, "
        f"ADR-0005).\n# index.html, manuscript.html, README.md and this file are generated.\n"
        f"{manifest}\n")

    postflight(out)
    print(f"\nBuilt {out}")
    print("NOT yet publishable: docs/ETHICS.md §5 checklist (IRB, consent/ToS, licence review, "
          "data-availability statement) is the gate and is a human determination.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
