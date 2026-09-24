"""Render the popular account into the showcase repository's `index.html`.

The landing page is generated, never hand-written. The popular article passed a methodology critic
and then had a round of factual corrections applied to it; a hand-maintained HTML copy would drift
from that gated text within a week and there would be no way to tell which of the two was wrong.
Generating means the Markdown stays the single source of the prose, and it means the generated page
is scanned by the disclosure gate along with everything else (`scripts/check_release_cells.py` reads
`.html`, and `build_public_release.py` runs the post-flight over the assembled tree).

Two rewrites are needed. Asset paths in the article are relative to `70-papers/popular/`, and at the
repository root they have to lose the `../../`. Videos are already raw `<video>` tags in the
Markdown, so the reader has to be told to pass raw HTML through rather than escape it.

    python scripts/build_showcase_site.py                     # writes ./index.html
    python scripts/build_showcase_site.py --out ../repo/index.html
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTICLE = ROOT / "70-papers" / "popular" / "dreams-arrow-of-time.md"

# Deliberately no external stylesheet, font or analytics script. A page about data that was never
# allowed to leave a machine should not make a third-party request when a stranger opens it.
CSS = """
:root {
  --ink: #1a1a1a; --mute: #5f6368; --rule: #e3e3e6;
  --bg: #ffffff; --card: #ffffff; --accent: #2f4f7f;
}
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font: 400 19px/1.65 Charter, Georgia, "Iowan Old Style", serif;
}
.wrap { max-width: 45rem; margin: 0 auto; padding: 0 1.25rem 6rem; }
header.masthead {
  border-bottom: 1px solid var(--rule); background: var(--card);
  margin-bottom: 3rem; padding: 3.5rem 1.25rem 2.5rem;
}
header.masthead .inner { max-width: 45rem; margin: 0 auto; }
.kicker {
  font: 600 12px/1.4 ui-sans-serif, -apple-system, "Segoe UI", sans-serif;
  letter-spacing: .13em; text-transform: uppercase; color: var(--accent); margin: 0 0 .9rem;
}
h1 { font-size: 2.45rem; line-height: 1.14; margin: 0 0 1rem; letter-spacing: -.02em; }
.standfirst { font-size: 1.19rem; color: var(--mute); margin: 0; max-width: 38rem; }
.byline {
  margin: 1.6rem 0 0; font: 400 15px/1.5 ui-sans-serif, -apple-system, sans-serif; color: var(--mute);
}
h2 {
  font-size: 1.55rem; line-height: 1.25; margin: 3.2rem 0 1rem; letter-spacing: -.01em;
  padding-top: .4rem;
}
h3 { font-size: 1.16rem; margin: 2.2rem 0 .6rem; }
p { margin: 0 0 1.25rem; }
a { color: var(--accent); text-decoration-thickness: 1px; text-underline-offset: 2px; }
hr { border: 0; border-top: 1px solid var(--rule); margin: 2.8rem 0; }
blockquote {
  margin: 1.6rem 0; padding: .2rem 0 .2rem 1.3rem;
  border-left: 3px solid var(--accent); color: var(--mute); font-style: italic;
}
code, pre {
  font: 400 .85em/1.5 ui-monospace, SFMono-Regular, Menlo, monospace;
  background: #f1f1ef; border-radius: 4px;
}
code { padding: .12em .38em; }
pre { padding: 1rem 1.1rem; overflow-x: auto; border: 1px solid var(--rule); }
pre code { background: none; padding: 0; }
figure { margin: 2.2rem 0; }
img, video {
  display: block; width: 100%; height: auto; border-radius: 6px;
  border: 0; background: var(--card);
}
figcaption, video + em, p > em:only-child {
  font: 400 15px/1.55 ui-sans-serif, -apple-system, sans-serif; color: var(--mute);
}
figcaption { margin-top: .7rem; }
table { border-collapse: collapse; width: 100%; margin: 1.8rem 0; font-size: .92rem; }
th, td { text-align: left; padding: .5rem .6rem; border-bottom: 1px solid var(--rule); }
th { font: 600 13px/1.4 ui-sans-serif, sans-serif; text-transform: uppercase; letter-spacing: .05em; }
ul, ol { margin: 0 0 1.25rem; padding-left: 1.4rem; }
li { margin-bottom: .45rem; }
footer.colophon {
  border-top: 1px solid var(--rule); margin-top: 4rem; padding-top: 1.6rem;
  font: 400 15px/1.6 ui-sans-serif, -apple-system, sans-serif; color: var(--mute);
}
@media (max-width: 34rem) {
  body { font-size: 18px; }
  h1 { font-size: 1.95rem; }
  header.masthead { padding: 2.5rem 1.1rem 2rem; }
}
"""

SHELL = """<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="{desc}">
<style>{css}</style>
<header class="masthead"><div class="inner">
  <p class="kicker">{kicker}</p>
  <h1>{title}</h1>
  <p class="standfirst">{desc}</p>
  <p class="byline">{byline}</p>
</div></header>
<main class="wrap">
{body}
<footer class="colophon">
{colophon}
</footer>
</main>
</html>
"""


def split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end < 0:
        return {}, text
    meta: dict[str, str] = {}
    for line in text[3:end].splitlines():
        if ":" in line and not line.startswith((" ", "-", "#")):
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip().strip('"').strip("'")
    return meta, text[end + 4:]


_LINK = re.compile(r"""(?x)
    (?P<md>\]\(\s*)(?P<mdpath>[^)\s]+)          # ](path)
  | (?P<attr>\b(?:src|href)\s*=\s*")(?P<apath>[^"]+)   # src="path" / href="path"
""")


def rebase_links(md: str, article_dir: Path, root: Path) -> tuple[str, int]:
    """Rewrite every relative link so it is correct from the tree root, where index.html sits.

    Resolved against the article's own directory rather than matched by prefix, because the article
    links to siblings (`../academic/dream-dynamics.md`) as well as to top-level directories, and a
    prefix list silently missed the siblings. Anything resolving outside the tree is left untouched
    so `--check-links` reports it instead of the rewrite hiding it.
    """
    n = 0

    def one(m: re.Match) -> str:
        nonlocal n
        head, path = (m.group("md"), m.group("mdpath")) if m.group("md") else \
                     (m.group("attr"), m.group("apath"))
        if path.startswith(("http://", "https://", "#", "mailto:", "/", "data:")):
            return m.group(0)
        bare, sep, frag = path.partition("#")
        try:
            new = (article_dir / bare).resolve().relative_to(root).as_posix()
        except ValueError:
            return m.group(0)
        n += 1
        return f"{head}{new}{sep}{frag}"

    return _LINK.sub(one, md), n


def check_links(html: str, tree: Path) -> list[str]:
    """Report every local href/src in the page that does not exist under `tree`."""
    refs = re.findall(r'(?:src|href)="([^"#?:]+)"', html)
    return sorted({r for r in refs if not r.startswith(("/", "#", "mailto")) and not (tree / r).exists()})


def drop_leading_h1(md: str) -> str:
    """The masthead already prints the title; a second copy in the body reads as a mistake."""
    return re.sub(r"\A\s*#\s+[^\n]+\n", "", md, count=1)


def pandoc_fragment(md: str) -> str:
    r = subprocess.run(
        ["pandoc", "-f", "markdown+raw_html-implicit_figures", "-t", "html5", "--wrap=none"],
        input=md, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"pandoc failed:\n{r.stderr[-2000:]}")
    if r.stderr.strip():
        print(f"      pandoc: {r.stderr.strip().splitlines()[0]}")
    return r.stdout


def build(article: Path = ARTICLE, out: Path | None = None) -> Path:
    out = out or (ROOT / "index.html")
    meta, body_md = split_frontmatter(article.read_text())
    body_md, n_assets = rebase_links(drop_leading_h1(body_md), article.parent.resolve(), ROOT)
    html = SHELL.format(
        css=CSS.strip(),
        title=meta.get("title", "Measuring a dreaming population"),
        desc=meta.get("subtitle", meta.get("description",
             "What 75,000 dream reports show about the temporal organization of a population's "
             "inner life — and what they do not.")),
        kicker=meta.get("kicker", "Dream dynamics"),
        byline=meta.get("byline", "Alexander V. Lebedev"),
        body=pandoc_fragment(body_md).strip(),
        colophon=(
            '<p>Generated from <code>70-papers/popular/dreams-arrow-of-time.md</code> by '
            '<code>scripts/build_showcase_site.py</code> — edit the Markdown, not this page.</p>\n'
            '<p>Prose, figures and videos are CC BY 4.0; code is MIT. No dream report text, user '
            'identifier or cell below the disclosure floor appears here or anywhere in this '
            'repository.</p>'),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"      index.html: {len(html) / 1024:.0f} KB, {n_assets} asset paths rebased")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None, help="output path (default: ./index.html)")
    ap.add_argument("--article", default=str(ARTICLE))
    ap.add_argument("--check-links", metavar="TREE",
                    help="verify every local link resolves inside TREE, and fail if not")
    a = ap.parse_args()
    out = build(Path(a.article), Path(a.out) if a.out else None)
    if a.check_links:
        broken = check_links(out.read_text(), Path(a.check_links).resolve())
        for b in broken:
            print(f"   BROKEN LINK  {b}", file=sys.stderr)
        if broken:
            raise SystemExit(f"{len(broken)} link(s) in index.html do not resolve in "
                             f"{a.check_links}")
        print("      all local links resolve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
