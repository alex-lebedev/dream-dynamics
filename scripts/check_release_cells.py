"""Enforce the small-cell disclosure rule on everything the repo actually releases.

docs/ETHICS.md §4 is categorical: do not publish any cell computed from fewer than 20 dreams or
fewer than 5 distinct users. `dream-dynamics.md` §2.6 and §6 both *assert* the release meets that
rule, which means the assertion has to be checkable — and until 2026-08-21 it was false, because
`personal_monthly.csv` (an N-of-1 series, cells down to n=1) and 33 cells of
`anomalous_affect_days.csv` (n=15) were git-tracked. Asserting compliance in prose is not
compliance; this script is what makes the sentence true.

Scope, and the mistake that made this paragraph necessary. The first version of this script scoped
itself to `git ls-files` — the *index*. That is the wrong question. The index describes what the
working copy tracks right now; "published" describes what someone who clones the repository can
read. Those diverged immediately: the four violating files were removed from the index with
`git rm --cached` and the deletion was left uncommitted, so the index was clean while `HEAD` still
served a `dreamseer_daily.csv` row computed from one dream by one user on an exact date. The check
passed and the violation was fully retrievable. A check whose scope is narrower than its claim is
worse than no check, because it launders the claim.

So this script now scans three scopes and fails on any of them:

  index    what `git ls-files` tracks — the working copy's own claim
  HEAD     what `git ls-tree -r HEAD` serves — what a plain clone checks out
  history  every blob reachable from any commit — what a clone actually *carries*

The third is the one that matters for an irreversible public push, and it is the one no amount of
`git rm` will satisfy: removing a file in a new commit leaves it in history forever, so clearing
this scope requires a history rewrite (`git filter-repo --invert-paths`). Pass `--scope` to narrow
the scan while iterating; the default is all three, and that is what must pass before a push.

Usage:
    python scripts/check_release_cells.py                  # exit 1 on any violation, all scopes
    python scripts/check_release_cells.py --list           # print every column read as a count
    python scripts/check_release_cells.py --scope index    # index only (fast, for iterating)
"""
from __future__ import annotations
import argparse
import io
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

MIN_REPORTS = 20
MIN_USERS = 5

# Columns that carry a per-cell report count or contributor count. Matched case-insensitively
# against the whole column name. Kept explicit rather than inferred: a regex loose enough to catch
# every count column also catches sample sizes of surrogates and bootstrap draws, which are not
# disclosure-relevant and would produce noise that trains the reader to ignore this check.
# `n_state`/`n_users_state` are per-*statistic* denominators: one released row can carry a raw mean
# over every report filed that day and a state estimate over the smaller subset with a within-person
# baseline. Gating only on the day's total volume releases the second below the floor, which is how
# a violation reached the working tree once already.
REPORT_COUNT_COLS = {"n_dreams", "n_reports", "n_docs", "n_documents", "n_state"}
USER_COUNT_COLS = {"n_users", "n_contributors", "n_authors", "n_dreamers", "n_users_state"}

# Counts that describe a resampling budget or a design dimension rather than a disclosed cell.
NOT_DISCLOSURE_COUNTS = {"n_surrogates", "n_perm", "n_draws", "n_boot", "n_events", "n_episodes",
                         "n_episodes_scored", "n_days", "n_cells", "n_seeds", "n_clusters",
                         "n_nights", "n_weeks", "n_crisis", "n_spikes"}

# A bare "n" is genuinely ambiguous across this repo: in `event_results.csv` it is a count of
# events, in `findings_ranked.csv` a count of weeks, in the old `personal_monthly.csv` a count of
# dreams. Treating it as a report count produces false failures, and failures a reader learns to
# dismiss are worse than no check. It is therefore surfaced for confirmation, never failed on.
AMBIGUOUS_COUNT_COLS = {"n", "n_obs"}

# Columns that identify a cell rather than report a statistic about it. Needed for the suppression
# test below: a row is only "suppressed" if nothing is left in it but its identity and its size.
KEY_COLS = {"lang", "date", "week", "month", "feature", "series", "component", "statistic",
            "n_userdays"}

RELEASE_DIRS = ("60-results/", "10-data/processed/")

AMBIGUOUS_HITS: list[str] = []


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout


def _is_release_csv(path: str) -> bool:
    return path.startswith(RELEASE_DIRS) and path.endswith(".csv")


def _is_release_artifact(path: str) -> bool:
    """Anything shipped under a release directory, whatever its file type.

    The distinction from `_is_release_csv` is the point of the inventory below. The version of this
    script that scanned CSVs only would print PASS while a violation sat in plain sight in a Markdown
    report, and one did: `personal_monthly.csv` was purged while `personal_n1_report.md`, carrying the
    same single contributor's yearly cells, stayed in the head commit. A check whose scope is narrower
    than the claim it licenses launders that claim, so every release artifact is now enumerated and
    anything the parser cannot read is accounted for by hand rather than skipped in silence.
    """
    return path.startswith(RELEASE_DIRS) and not path.endswith(".gitkeep")


# `.html` earns its place here for the same reason `.md` did. The public showcase renders its
# walkthrough to HTML, and that page quotes percentages and sample sizes; a scanner that skipped the
# one file most likely to be read would launder exactly the claim it exists to license.
TEXT_EXT = (".md", ".json", ".txt", ".tsv", ".graphml", ".yaml", ".yml", ".log", ".html")
FIGURE_EXT = (".png", ".pdf", ".svg", ".jpg", ".jpeg")

# What actually marks the violations found so far, which is NOT a small report count. Every N-of-1
# leak caught here had a *large* report count and one contributor: the purged monthly table (754
# dreams), the `grand_unified.md` line ("You (N=1) ... n=754"), the replication report's N=1 section
# (~356 dreams). Keying the tripwire on reports below twenty would have missed all three. So the
# primary signal is the single-contributor label, with sub-floor counts as a secondary catch.
# The `(?![\d,])` guards are the same ones the count patterns carry, and for the same reason: without
# them "n = 1,500" matches the bare `N=1` alternative on its thousands separator, so any prose citing
# a four-figure sample size reads as an N-of-1 cell. "N=10" was never affected (no word boundary
# after the 1), and "N=1." at the end of a sentence still fires.
_N_OF_1 = re.compile(r"(?i)(?:\bN\s*[=＝]\s*1\b(?![\d,])|\bN-of-1\b|\bpersonal\s+N\s*=\s*1\b(?![\d,])|\byou\s*\(N\s*=\s*1\)|\bsingle[- ]contributor\b)")

# A count *labelled* as one and below its floor. Deliberately narrow, so a probability or a year is
# not a hit; the negative lookahead stops "n=5,884" matching on its thousands separator.
_COUNT_USERS = re.compile(
    r"""(?ix) (?: \b n_(?:users|contributors|authors|dreamers) \b | " n_(?:users|contributors) " )
        \s* [=:]? \s* (\d{1,3}) (?![\d,])""")
_COUNT_REPORTS = re.compile(
    r"""(?ix) (?: \b n_(?:dreams|reports|docs) \b | " n_(?:dreams|reports) " )
        \s* [=:]? \s* (\d{1,3}) (?![\d,])""")

# A bare `n=` in prose is as ambiguous as a bare `n` column, and for the same reason: across these
# reports it counts months, days, weeks, spikes and subset sizes far more often than it counts
# dreams. Failing on it produces the noise that trains a reader to skip the check, so it is surfaced
# for confirmation on the same terms as its tabular counterpart.
_COUNT_BARE = re.compile(r"""(?ix) \b n \s* = \s* (\d{1,3}) (?![\d,])""")

# Non-CSV artifacts a human has read and cleared. Extension alone is not a defence for text; keep the
# reason with the entry so the clearance is reviewable rather than inherited.
REVIEWED_NON_CSV_FILES: dict[str, str] = {}


def _scan_text(label: str, raw: str) -> list[str]:
    """Tripwire over one text-like release artifact. Returns failure strings."""
    out = []
    m = _N_OF_1.search(raw)
    if m:
        ctx = " ".join(raw[max(0, m.start() - 70):m.end() + 70].split())
        out.append(f"{label}: single-contributor (N-of-1) cell — ...{ctx}...")
    for rx, floor, what in ((_COUNT_USERS, MIN_USERS, "contributor"),
                            (_COUNT_REPORTS, MIN_REPORTS, "report")):
        for hit in rx.finditer(raw):
            k = int(hit.group(1))
            if k < floor:
                ctx = " ".join(raw[max(0, hit.start() - 70):hit.end() + 30].split())
                out.append(f"{label}: {what} count of {k} below the {floor}-{what} floor — ...{ctx}...")
                break
    for hit in _COUNT_BARE.finditer(raw):
        if int(hit.group(1)) < MIN_REPORTS:
            ctx = " ".join(raw[max(0, hit.start() - 60):hit.end() + 20].split())
            AMBIGUOUS_HITS.append(f"{label}: bare 'n={hit.group(1)}' in prose — confirm it counts "
                                  f"months/days/events and not reports — ...{ctx}...")
            break
    return out


def non_csv_inventory(scope: str) -> list[str]:
    """Enumerate every non-CSV release artifact and tripwire-scan the readable ones.

    Four scopes, not three. The working tree is included even though it is not yet published, because
    that is where the next violation is staged: an untracked report sitting in a release directory is
    one `git add -A` away from being a disclosure, and catching it there is the only cheap moment.
    Figures clear by type; text-like artifacts are scanned; anything of neither kind is listed for a
    human, so the limit travels with the result instead of having to be inferred.
    """
    seen: dict[str, tuple[str, str, bool]] = {}     # key -> (label, source, read_from_git)

    if scope in ("worktree", "all"):
        # Untracked but NOT gitignored: exactly the set that a `git add -A` would publish. Ignored
        # files are excluded because they cannot reach a reader, and including them buries the
        # signal under every cached parquet in the tree.
        for p in _git("ls-files", "--others", "--exclude-standard").splitlines():
            if _is_release_artifact(p) and not p.endswith(".csv"):
                seen[f"worktree:{p}"] = (f"worktree:{p}", p, False)
    if scope in ("index", "all"):
        for p in _git("ls-files").splitlines():
            if _is_release_artifact(p) and not p.endswith(".csv"):
                seen[p] = (p, p, False)
    if scope in ("head", "all"):
        for line in _git("ls-tree", "-r", "HEAD").splitlines():
            meta, _, path = line.partition("\t")
            parts = meta.split()
            if len(parts) >= 3 and parts[1] == "blob" and _is_release_artifact(path) \
                    and not path.endswith(".csv"):
                seen.setdefault(f"HEAD:{path}", (f"HEAD:{path}", parts[2], True))
    if scope in ("history", "all"):
        for line in _git("rev-list", "--objects", "--all").splitlines():
            sha, _, path = line.partition(" ")
            path = path.strip()
            if path and _is_release_artifact(path) and not path.endswith(".csv") \
                    and Path(path).suffix:              # skip tree objects, which have no extension
                seen.setdefault(f"history:{sha[:8]}:{path}", (f"history:{path}", sha, True))

    figures = {k for k, v in seen.items() if v[0].endswith(FIGURE_EXT)}
    scannable = {k: v for k, v in seen.items() if v[0].endswith(TEXT_EXT)}
    unreadable = [k for k in seen if k not in figures and k not in scannable]

    problems: list[str] = []
    for key in sorted(scannable):
        label, source, from_git = scannable[key]
        if label in REVIEWED_NON_CSV_FILES:
            continue
        try:
            raw = _git("cat-file", "-p", source) if from_git else Path(source).read_text()
        except Exception:
            unreadable.append(key)
            continue
        problems += _scan_text(label, raw)

    print(f"\n[scan] non-CSV release artifacts: {len(figures)} figure(s) cleared by type, "
          f"{len(scannable)} text artifact(s) tripwire-scanned for single-contributor cells and "
          f"sub-floor counts in prose or JSON, {len(unreadable)} of neither kind.")
    for k in sorted(unreadable):
        print(f"    cannot read, clear by hand: {seen[k][0]}")
    return problems


def worktree_files() -> list[tuple[str, str]]:
    """(label, path) for untracked, non-ignored CSVs in the release paths.

    Not yet published, and deliberately included anyway: an untracked table sitting in a release
    directory is one `git add -A` away from being a disclosure, and that is the only cheap moment to
    catch it. Ignored files are excluded because they cannot reach a reader.
    """
    return [(f"worktree:{p}", p)
            for p in _git("ls-files", "--others", "--exclude-standard").splitlines()
            if _is_release_csv(p)]


def index_files() -> list[tuple[str, str]]:
    """(label, path) for CSVs in the index; read from the working tree."""
    return [(p, p) for p in _git("ls-files").splitlines() if _is_release_csv(p)]


def head_blobs() -> list[tuple[str, str]]:
    """(label, blob-sha) for CSVs in HEAD's tree — what a plain clone checks out."""
    out = []
    for line in _git("ls-tree", "-r", "HEAD").splitlines():
        meta, _, path = line.partition("\t")
        parts = meta.split()
        if len(parts) >= 3 and parts[1] == "blob" and _is_release_csv(path):
            out.append((f"HEAD:{path}", parts[2]))
    return out


def history_blobs() -> list[tuple[str, str]]:
    """(label, blob-sha) for every distinct CSV blob reachable from any commit.

    This is the scope that matters before an irreversible push: `git rm` in a new commit does not
    remove a blob from history, so a clone carries every version that ever existed. Deduplicated by
    blob sha, since the same content recurring across commits is one disclosure, not many.
    """
    out = _git("rev-list", "--objects", "--all")
    seen: dict[str, str] = {}
    for line in out.splitlines():
        sha, _, path = line.partition(" ")
        if path and _is_release_csv(path.strip()) and sha not in seen:
            seen[sha] = path.strip()
    return [(f"history:{p}", sha) for sha, p in seen.items()]


def _read(source: str, from_git: bool) -> pd.DataFrame:
    if not from_git:
        return pd.read_csv(source)
    blob = subprocess.run(["git", "cat-file", "blob", source],
                          capture_output=True, check=True).stdout
    return pd.read_csv(io.BytesIO(blob))


def check(label: str, source: str, from_git: bool, show: bool) -> list[str]:
    try:
        df = _read(source, from_git)
    except Exception as e:  # a table we cannot parse is a table we cannot clear
        return [f"{label}: UNREADABLE ({e})"]

    problems = []
    for col in df.columns:
        key = str(col).strip().lower()
        if key in NOT_DISCLOSURE_COUNTS:
            continue
        if key in AMBIGUOUS_COUNT_COLS:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if not vals.empty and (vals < MIN_REPORTS).any():
                AMBIGUOUS_HITS.append(
                    f"{label}: column '{col}' has values below {MIN_REPORTS} "
                    f"(min {vals.min():g}) — confirm it counts events/weeks and not reports")
            continue
        if key in REPORT_COUNT_COLS:
            floor, what = MIN_REPORTS, "reports"
        elif key in USER_COUNT_COLS:
            floor, what = MIN_USERS, "users"
        else:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if vals.empty:
            continue
        if show:
            print(f"    {label}: column '{col}' read as a {what} count "
                  f"(min={vals.min():g}, floor={floor})")
        thin = vals.index[vals < floor]
        if not len(thin):
            continue

        # A sub-floor count is not automatically a disclosure. The released aggregates deliberately
        # retain their denominator on every row while blanking the statistics, because a count over
        # the population is not a disclosive attribute and keeping it is what makes the suppression
        # auditable (manuscript §6). So the question is not "is this cell thin" but "is anything
        # still readable in it" — which makes this a stronger check than the flat count test it
        # replaces, since it verifies the suppression happened instead of trusting that it did.
        stat_cols = [c for c in df.columns
                     if str(c).strip().lower() not in (KEY_COLS | REPORT_COUNT_COLS
                                                       | USER_COUNT_COLS | AMBIGUOUS_COUNT_COLS
                                                       | NOT_DISCLOSURE_COUNTS)]
        if stat_cols:
            leaked = int(df.loc[thin, stat_cols].notna().any(axis=1).sum())
        else:
            leaked = 0  # nothing but identity and size in the row: nothing to disclose
        if leaked:
            problems.append(f"{label}: {leaked} of {len(vals)} cells have {col} < {floor} "
                            f"(min {vals.min():g}) AND carry unsuppressed statistics — below the "
                            f"{floor}-{what} disclosure floor")
        elif show:
            print(f"      -> {len(thin)} sub-floor rows, all statistics suppressed (ok)")
    return problems


# The tripwire's sensitivity, as a runnable check rather than a reported one. The failure that made
# this necessary was a capability described in prose that had silently ceased to exist in the code,
# so a claim about what this scanner catches should be executable by anyone who reads the claim.
SELF_TEST: list[tuple[str, str, bool]] = [
    ("grand_unified line (1 contributor, 754 dreams)",
     "- You (N=1): -0.107 +/- 0.025 (n=754)", True),
    ("personal_n1_report yearly cell",
     "| 2021 | -0.19 | n_dreams=14 | n_users=1 |", True),
    ("dataset_replication section header",
     "## Personal N=1 (monthly model-sentiment; very low power, ~356 dreams)", True),
    ("a legitimate corpus aggregate",
     "- DreamSeer EN: -0.246 +/- 0.004 (n=24,311), n_users=3182", False),
    ("a legitimate month count",
     "- RU M (n=14): r=+0.245 (phase-surrogate p=0.750)", False),
    ("a legitimate subset-size label",
     "## Gate sensitivity (EN) - is the effect real or an n=3 subset artifact?", False),
    # Regression: the N-of-1 alternative used to match "n = 1,500" on its thousands separator, which
    # failed the release build on a video note that was reporting a 1,500-report aggregate.
    ("a four-figure sample size written with a separator",
     "- Distance to the held-out arc falls from 0.076 at n = 20 to 0.0187 at n = 1,500", False),
    ("an N-of-1 cell that must still fire when the sentence ends",
     "Monthly model-sentiment for the operator's own reports, N=1.", True),
    # The showcase page is HTML, so the same leak has to be caught through markup as through prose.
    ("an N-of-1 cell hidden inside HTML markup",
     "<p>Personal series for a single diarist (<em>N=1</em>), 754 dreams.</p>", True),
]


def self_test() -> int:
    print("[self-test] does the tripwire fire on the violations it is credited with catching,\n"
          "            and stay silent on the legitimate cells they sit beside?\n")
    bad = 0
    for name, text, should_fire in SELF_TEST:
        AMBIGUOUS_HITS.clear()
        fired = bool(_scan_text("self-test", text))
        ok = fired == should_fire
        bad += not ok
        want = "must fire" if should_fire else "must stay silent"
        got = "fires" if fired else ("ambiguous only" if AMBIGUOUS_HITS else "silent")
        print(f"    {'ok ' if ok else 'FAIL'} {name:44s} ({want:16s}) -> {got}")
    print(f"\n[self-test] {'PASS' if not bad else f'{bad} case(s) FAILED'}")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="print every column interpreted as a count")
    ap.add_argument("--scope", choices=["worktree", "index", "head", "history", "all"],
                    default="all")
    ap.add_argument("--self-test", action="store_true",
                    help="check the non-CSV tripwire against known violations and known-clean cells")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    scopes = []
    if args.scope in ("worktree", "all"):
        scopes.append(("worktree (untracked, not yet published)", worktree_files(), False))
    if args.scope in ("index", "all"):
        scopes.append(("index", index_files(), False))
    if args.scope in ("head", "all"):
        scopes.append(("HEAD", head_blobs(), True))
    if args.scope in ("history", "all"):
        scopes.append(("history", history_blobs(), True))

    problems: list[str] = []
    for name, items, from_git in scopes:
        print(f"[scan] {name}: {len(items)} CSVs under {', '.join(RELEASE_DIRS)}")
        for label, source in sorted(items):
            problems += check(label, source, from_git, args.list)

    # A released table with no count column at all is not automatically fine; it may be an
    # aggregate whose denominator was dropped. Report those so the omission is a decision. Scoped
    # to the index, since that is the set a reader is asked to interpret.
    idx = [p for _, p in index_files()]
    countless = [f for f in sorted(idx)
                 if not ({str(c).strip().lower() for c in pd.read_csv(f, nrows=0).columns}
                         & (REPORT_COUNT_COLS | USER_COUNT_COLS))]
    if countless:
        print(f"\n[note] {len(countless)} of {len(idx)} tracked tables carry no per-cell count "
              f"column, so this check cannot clear them; confirm each is a statistic over the full "
              f"corpus rather than a thin subgroup:")
        for f in countless:
            print(f"    {f}")

    problems += non_csv_inventory(args.scope)

    if AMBIGUOUS_HITS:
        print(f"\n[confirm] {len(AMBIGUOUS_HITS)} ambiguous count column(s), not failed on:")
        for a in sorted(set(AMBIGUOUS_HITS)):
            print(f"    {a}")

    if problems:
        print(f"\n[FAIL] {len(problems)} disclosure violation(s):")
        for p in problems:
            print(f"    {p}")
        print("\nA violation under 'history' is not fixable by deleting the file in a new commit; "
              "it needs `git filter-repo --invert-paths --path <p>` before any push.")
        return 1
    print(f"\n[PASS] every counted cell in the working tree, the index, HEAD and reachable history "
          f"meets n>={MIN_REPORTS} reports and n>={MIN_USERS} users.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
