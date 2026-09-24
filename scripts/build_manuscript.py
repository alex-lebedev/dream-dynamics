"""Render `dream-dynamics.md` to .docx with submission-appropriate styling.

Three defaults of pandoc's docx writer are wrong for a manuscript going to a journal, and all
three are invisible until someone opens the file in Word:

1.  HEADINGS ARE BLUE. Pandoc's built-in reference document styles Title and Heading1-6 with the
    theme's accent colour (`4F81BD`, and `345A8A` for the Title). No journal wants a two-colour
    manuscript. This script derives a reference document from pandoc's own default and sets those
    runs to automatic colour, changing nothing else, so the result stays a valid pandoc reference
    doc rather than a hand-built template that drifts from it.
2.  EVERY HEADING CARRIES A BOOKMARK. Pandoc assigns each heading an identifier and writes it as a
    Word bookmark. Word draws bookmarks as grey square brackets when *Show bookmarks* is on, so
    the document appears to have a stray `[` before every title. They are non-printing and would
    never reach a PDF, but they are alarming on screen and there is no reason to emit them for a
    format nothing links into: `-auto_identifiers` drops them, and `link-citations=false` drops
    the citation-to-bibliography links that would otherwise need their own.
3.  Section numbering is left alone deliberately. The manuscript writes its own numbers into the
    heading text ("# 1. Introduction"), so `--number-sections` would double them.

A fourth problem is not pandoc's fault and is the most consequential: the manuscript embeds its
figures with Obsidian's wiki syntax, `![[60-results/...png]]`, which is not markdown. Pandoc passes
it through as literal text, so every earlier .docx export contained **no images at all** — twelve
figures rendered as bracketed paths. Rewriting the source to standard `![](...)` would fix the export
and break the preview in Obsidian, where the manuscript is actually written, so the rewrite happens
at build time on a temporary copy instead. The resource path includes the repository root because
those embeds are root-relative while the manuscript is two directories down.

The reference document is written to `70-papers/academic/reference.docx` and regenerated only
when absent or when --refresh is passed, so a reference doc that has since been hand-edited in
Word (to set a journal's font, say) is not silently overwritten.

    python3 scripts/build_manuscript.py                 # -> dream-dynamics.docx
    python3 scripts/build_manuscript.py --refresh        # rebuild reference.docx from pandoc's
    python3 scripts/build_manuscript.py --keep-bookmarks # emit heading anchors after all
    python3 scripts/build_manuscript.py --html out.html  # -> citation-resolved HTML (showcase repo)
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "70-papers" / "academic"
MANUSCRIPT = PAPER / "dream-dynamics.md"
REFDOC = PAPER / "reference.docx"
OUT = PAPER / "dream-dynamics.docx"

# Styles whose colour pandoc's default reference doc sets to the theme accent.
COLOURED = ("Title", "Subtitle", "Author", "Date", *(f"Heading{i}" for i in range(1, 10)))


def pandoc(*args: str) -> bytes:
    """Run pandoc, surfacing its stderr rather than a bare non-zero exit."""
    p = subprocess.run(["pandoc", *args], capture_output=True)
    if p.returncode:
        sys.exit(f"pandoc failed:\n{p.stderr.decode(errors='replace')}")
    return p.stdout


def decolour(styles: str) -> tuple[str, list[str]]:
    """Set every accent-coloured heading run to automatic colour; report which changed."""
    changed: list[str] = []

    def one(m: re.Match) -> str:
        sid, body = m.group(1), m.group(0)
        if sid not in COLOURED or "<w:color" not in body:
            return body
        new = re.sub(r"<w:color\b[^>]*/>", '<w:color w:val="auto"/>', body)
        if new != body:
            changed.append(sid)
        return new

    out = re.sub(r"<w:style\b[^>]*w:styleId=\"([^\"]+)\"[^>]*>.*?</w:style>", one, styles,
                 flags=re.S)
    return out, changed


def strip_bookmarks(path: Path) -> int:
    """Remove every Word bookmark from a .docx, returning how many were dropped.

    `-auto_identifiers` suppresses heading anchors, but citeproc gives each bibliography entry an
    explicit `ref-<key>` id and pandoc writes those as bookmarks too. With citation linking off
    nothing points at them, so they are pure noise — and noise Word renders as a grey '[' in the
    reference list. Starts and ends are removed as matched pairs, keyed on the ids of the starts
    that were dropped, so the file cannot be left with a dangling bookmarkEnd.
    """
    with zipfile.ZipFile(path) as zin:
        names = zin.namelist()
        blobs = {n: zin.read(n) for n in names}
    doc = blobs["word/document.xml"].decode()

    dropped: set[str] = set()

    def start(m: re.Match) -> str:
        dropped.add(re.search(r'w:id="([^"]+)"', m.group(0)).group(1))
        return ""

    doc = re.sub(r"<w:bookmarkStart\b[^>]*/>", start, doc)
    doc = re.sub(r"<w:bookmarkEnd\b[^>]*/>",
                 lambda m: "" if re.search(r'w:id="([^"]+)"', m.group(0)).group(1) in dropped
                 else m.group(0), doc)
    blobs["word/document.xml"] = doc.encode()
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zout:
        for n in names:
            zout.writestr(n, blobs[n])
    return len(dropped)


def unwiki(text: str) -> tuple[str, int]:
    """Rewrite Obsidian embeds `![[path]]` / `![[path|alt]]` as markdown images `![alt](path)`.

    Alt text is left empty when Obsidian supplied none, which keeps pandoc's `implicit_figures`
    from generating a caption: the manuscript writes its own bold "Figure N." captions as the
    following paragraph, and an auto-caption would duplicate them.
    """
    n = 0

    def one(m: re.Match) -> str:
        nonlocal n
        n += 1
        target = m.group(1).strip()
        path, _, alt = target.partition("|")
        return f"![{alt.strip()}]({path.strip()})"

    return re.sub(r"!\[\[([^\]]+)\]\]", one, text), n


def build_refdoc() -> None:
    """Derive a reference doc from pandoc's default, with headings in a single colour."""
    default = REFDOC.with_suffix(".pandoc-default.docx")
    default.write_bytes(pandoc("--print-default-data-file", "reference.docx"))
    with zipfile.ZipFile(default) as zin:
        names = zin.namelist()
        blobs = {n: zin.read(n) for n in names}
    styles, changed = decolour(blobs["word/styles.xml"].decode())
    blobs["word/styles.xml"] = styles.encode()
    with zipfile.ZipFile(REFDOC, "w", zipfile.ZIP_DEFLATED) as zout:
        for n in names:                                   # preserve the original entry order
            zout.writestr(n, blobs[n])
    default.unlink()
    print(f"[docx] wrote {REFDOC.relative_to(ROOT)}; single-coloured: {', '.join(changed)}")


def render_html(out: Path) -> Path:
    """Render the manuscript to one self-contained HTML file with citations resolved.

    The showcase repository ships this instead of the .docx. The .docx is 15.6 MB of embedded
    figures, GitHub will not preview it, and it is a build product of the Markdown anyway. HTML
    renders the bibliography through the same citeproc path as the .docx, opens in any browser, and
    is diffable. Figures are referenced rather than embedded because the repository already ships
    them, and the manuscript's wiki-embeds are root-relative, so this must be written at the root of
    the tree for those paths to resolve.
    """
    src, n_fig = unwiki(MANUSCRIPT.read_text())
    tmp = PAPER / ".dream-dynamics.html.md"
    tmp.write_text(src)
    try:
        pandoc(str(tmp), "--citeproc", "-f", "markdown", "-t", "html5", "--standalone",
               "--metadata", "title=Measuring a dreaming population",
               f"--resource-path={PAPER}{os.pathsep}{ROOT}",
               f"--bibliography={PAPER / 'references-seed.bib'}",
               f"--csl={PAPER / 'nature-brackets.csl'}", "-o", str(out))
    finally:
        tmp.unlink(missing_ok=True)
    html = out.read_text()
    print(f"      manuscript.html: {len(html) / 1024:.0f} KB, {n_fig} figures referenced, "
          f"{html.count('csl-entry')} bibliography entries")
    return out


def main() -> None:
    if "--html" in sys.argv:
        i = sys.argv.index("--html")
        render_html(Path(sys.argv[i + 1]).resolve())
        return
    if "--refresh" in sys.argv or not REFDOC.exists():
        build_refdoc()
    fmt = "markdown" if "--keep-bookmarks" in sys.argv else "markdown-auto_identifiers"
    # The bibliography and style are passed here rather than declared in the manuscript's YAML,
    # which is a concession to the Obsidian reference-list plugin: it resolves a frontmatter
    # `bibliography` against the *vault root* rather than the note, and — in `getReferenceList`,
    # outside any try/except — throws when that path does not exist, which aborts the whole render
    # and leaves the sidebar claiming the document has no citations. A note with no such
    # frontmatter takes the plugin's own configured paths instead and renders correctly. Passing
    # them as arguments also makes the build independent of the working directory, which the
    # relative frontmatter was not.
    src, n_fig = unwiki(MANUSCRIPT.read_text())
    # Written beside the manuscript so that relative paths and the resource path behave identically.
    tmp = PAPER / ".dream-dynamics.build.md"
    tmp.write_text(src)
    args = [str(tmp), "--citeproc", "-f", fmt, f"--resource-path={PAPER}{os.pathsep}{ROOT}",
            f"--bibliography={PAPER / 'references-seed.bib'}",
            f"--csl={PAPER / 'nature-brackets.csl'}",
            f"--reference-doc={REFDOC}", "-o", str(OUT)]
    if "--keep-bookmarks" not in sys.argv:
        args += ["-M", "link-citations=false"]
    try:
        pandoc(*args)
    finally:
        tmp.unlink(missing_ok=True)

    if "--keep-bookmarks" not in sys.argv:
        gone = strip_bookmarks(OUT)
    with zipfile.ZipFile(OUT) as z:
        doc = z.read("word/document.xml").decode()
    marks = doc.count("<w:bookmarkStart")
    with zipfile.ZipFile(OUT) as z:
        images = [n for n in z.namelist() if n.startswith("word/media/")]
    print(f"[docx] wrote {OUT.relative_to(ROOT)} — "
          f"{len(re.findall(r'w:val=.Heading1.', doc))} top-level headings, "
          f"{n_fig} wiki-embeds rewritten, {len(images)} images embedded, "
          f"{marks} bookmarks" + (f" ({gone} stripped)" if "--keep-bookmarks" not in sys.argv
                                  else ""))
    if n_fig and not images:
        print("[docx] WARNING: embeds were rewritten but no image reached the document — "
              "check that the figure paths resolve from the repository root")
    if marks:
        print("[docx] note: Word draws bookmarks as grey '[' brackets when Show bookmarks is on")


if __name__ == "__main__":
    main()
