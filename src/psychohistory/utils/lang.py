"""Language tagging for dream text.

Strategy: Cyrillic script is a reliable, cheap signal for Russian. Otherwise use
`langdetect` if installed; fall back to a non-ASCII-ratio heuristic (en vs other).
We only need a coarse EN / RU / OTHER split for stratification (see docs/METHODS.md).
"""
from __future__ import annotations
import re

_CYR = re.compile(r"[\u0400-\u04FF]")
_NONASCII = re.compile(r"[^\x00-\x7F]")

try:  # optional, better accuracy for Latin-script languages
    from langdetect import detect as _ld_detect, DetectorFactory

    DetectorFactory.seed = 0
    _HAVE_LD = True
except Exception:  # pragma: no cover
    _HAVE_LD = False


def detect_language(text: str) -> str:
    """Return one of 'en', 'ru', 'other', or 'unknown'."""
    if not isinstance(text, str) or not text.strip():
        return "unknown"
    letters = [c for c in text if c.isalpha()]
    if letters:
        cyr_ratio = sum(1 for c in letters if _CYR.match(c)) / len(letters)
        if cyr_ratio > 0.20:
            return "ru"
    if _HAVE_LD and len(text) >= 20:
        try:
            code = _ld_detect(text)
            if code == "en":
                return "en"
            if code == "ru":
                return "ru"
            return "other"
        except Exception:
            pass
    nonascii_ratio = len(_NONASCII.findall(text)) / max(len(text), 1)
    return "other" if nonascii_ratio > 0.15 else "en"


def backend() -> str:
    return "langdetect" if _HAVE_LD else "heuristic(cyrillic+nonascii)"
