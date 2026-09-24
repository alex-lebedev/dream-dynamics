"""Declared theme lexicons for content-correspondence tests.

Defined here, once, so that every analysis scores the same axis the same way
(METHODS section 1: "Every feature is defined once, in code, with a
pre-registered formula"). Nothing in this module is tuned against an outcome.

The partition matters for the COVID content-correspondence test and is fixed
before estimation:

``EXPLICIT``
    Vocabulary that barely existed as a topic before 2020. A rise here is
    *expected* and carries no evidence about dream content — it measures
    topical availability and post narration. It is used two ways: as a
    manipulation check that the interrupted-time-series design can detect a
    real step, and as a screen — posts containing any explicit term are
    dropped from the primary arm, so the latent axes cannot be moved by people
    simply mentioning the pandemic.

``LATENT``
    Themes whose vocabulary was fully available before 2020. These carry the
    actual test: did dreams acquire more contagion / illness / contamination /
    death / mask content?

``COMPARATOR``
    Generic anxiety. This is the axis the repo's earlier COVID test used,
    conflated with the health terms; kept separate here so the earlier result
    can be reproduced and decomposed.

``NEUTRAL``
    Negative controls. A design that moves these has a composition or
    measurement problem, not a finding.

All groups are non-capturing: the patterns are used only for presence tests.
"""
from __future__ import annotations

import hashlib
import re

# --- vocabulary that did not exist as a topic before 2020 -------------------
EXPLICIT: dict[str, str] = {
    "explicit_pandemic": (
        r"covid|corona[ -]?virus|\bcorona\b|pandemic|quarantin\w*|lock[ -]?down|"
        r"social[ -]distanc\w*|self[ -]isolat\w*|\bppe\b|shelter[ -]in[ -]place"
    ),
}

# --- the actual test: themes whose words were available before 2020 ---------
LATENT: dict[str, str] = {
    "contagion": (
        r"\bvirus(?:es)?\b|\bviral\b|infect\w*|contagio\w*|epidemic|\bplagued?s?\b|"
        r"outbreak|\bgerms?\b|bacteri\w*"
    ),
    "illness": (
        r"\bsick(?:ness)?\b|\bill(?:ness)?(?:es)?\b|\bdiseased?s?\b|fever|cough\w*|"
        r"\bsymptoms?\b|\bhospitals?\b|\bdoctors?\b|\bnurses?\b|medicine|medication|"
        r"\bventilators?\b|\bicu\b|ambulance|surgery|cancer|diagnos\w*"
    ),
    "contamination": (
        r"\bdirty\b|\bfilthy?\b|unclean|poison\w*|\btoxic\b|pollut\w*|"
        r"\brot(?:ten|ting)?\b|decay\w*|sewage|\bmou?ldy?\b|putrid|\bvomit\w*|"
        r"\bgrime\b|\bstench\b"
    ),
    "death": (
        r"\bdied?s?\b|\bdying\b|\bdead\b|\bdeaths?\b|\bcorpses?\b|cadaver|\bfunerals?\b|"
        r"\bgrave(?:yard)?s?\b|cemeter\w*|\bcoffins?\b|\bcaskets?\b|\bburied\b|\bburial\b|"
        r"mourn\w*|morgue|\bkill(?:ed|ing|s)?\b|murder\w*"
    ),
    "mask": r"\bmasks?\b|\bmasked\b|face[ -]cover\w*",
}

# --- the axis the earlier COVID test used, with health terms removed -------
COMPARATOR: dict[str, str] = {
    "anxiety": (
        r"anxious|anxiety|\bscared\b|afraid|terrif\w*|\bpanic\w*|\bchased?s?\b|"
        r"\bchasing\b|\btrapped\b|frightened|\bdread\w*"
    ),
}

# --- negative controls: these must not move --------------------------------
NEUTRAL: dict[str, str] = {
    "food": r"\bfood\b|\beat(?:ing|s)?\b|\bate\b|\bmeals?\b|hungry|restaurant|kitchen",
    "family": (
        r"\bmother\b|\bfather\b|\bmom\b|\bmum\b|\bdad\b|\bsisters?\b|\bbrothers?\b|"
        r"\bfamily\b"
    ),
    "animal": r"\bdogs?\b|\bcats?\b|\banimals?\b|\bbirds?\b|\bhorses?\b|\bsnakes?\b|\bspiders?\b",
    "flying": r"\bfly(?:ing)?\b|\bflew\b|airplane|\bplanes?\b|\baircraft\b",
    "vehicle": r"\bcars?\b|\bdriv(?:e|ing|er)\b|\bdrove\b|\bbus\b|\btrains?\b",
    "school": r"\bschool\b|\bclass(?:room)?\b|\bteachers?\b|\bexams?\b|\bhomework\b",
    "water": r"\bwater\b|\bocean\b|\bsea\b|\briver\b|\bswim(?:ming)?\b|\blake\b|\bflood\w*",
}

#: Declared families, in the order they are reported. The family a test belongs
#: to fixes its multiplicity correction; the boundaries are set here rather than
#: after seeing any p-value.
FAMILIES: dict[str, dict[str, str]] = {
    "explicit": EXPLICIT,
    "latent": LATENT,
    "comparator": COMPARATOR,
    "neutral": NEUTRAL,
}

#: Every axis name, flattened.
ALL_AXES: dict[str, str] = {k: v for fam in FAMILIES.values() for k, v in fam.items()}


def version() -> str:
    """Short hash of the lexicon definitions, for cache invalidation."""
    blob = "|".join(f"{k}={v}" for k, v in sorted(ALL_AXES.items()))
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def compiled(axes: dict[str, str] | None = None) -> dict[str, re.Pattern]:
    """Case-insensitive compiled patterns for the requested axes (default: all)."""
    return {k: re.compile(v, re.I) for k, v in (axes or ALL_AXES).items()}


def family_of(axis: str) -> str:
    """Which declared family an axis belongs to."""
    for fam, members in FAMILIES.items():
        if axis in members:
            return fam
    raise KeyError(axis)


def score(texts, axes: dict[str, str] | None = None):
    """Binary presence of each axis in each text.

    Presence rather than count: counts are dominated by report length, and
    length itself drifts across these corpora.

    Accepts a pandas Series (vectorised, much faster) or any iterable of str.
    Returns a DataFrame for a Series input, a dict of lists otherwise.
    """
    pats = compiled(axes)
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover
        pd = None
    if pd is not None and isinstance(texts, pd.Series):
        return pd.DataFrame(
            {name: texts.str.contains(pat).astype("int8") for name, pat in pats.items()},
            index=texts.index,
        )
    texts = list(texts)
    return {name: [1 if pat.search(t) else 0 for t in texts] for name, pat in pats.items()}
