"""Calendars as natural experiments: the institutional week as a driver of nocturnal cognition.

The manuscript treats the weekday almost everywhere as a nuisance covariate and once, briefly, as a
result. Conceptually those are different objects. News reaches a dreamer through attention; the
working week reaches a dreamer through sleep — its timing, its duration, the debt accumulated across
it, the alarm that ends it. Social-zeitgeber accounts of mood make exactly this claim, that socially
imposed schedules entrain biological rhythms, and weekday-to-weekend shifts in sleep timing are
among the largest and most reliably measured effects in population sleep data. If a dreaming
population is sensitive to macro-social structure at all, that is the channel with the strongest
prior, and it is testable on the calendar alone.

This module builds the calendar regressors: the weekday profile, whether the *night* preceded a
free day or a working day, public holidays, holiday eves, daylight-saving transitions, and the
school-summer period. Everything here is a deterministic function of the date, so no external data
source and no attention proxy enters the analysis.

    from psychohistory.dreams.socialtime import social_design, FREE_NIGHT_NOTE
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Reports are filed on the morning after the dream, so a report dated Saturday describes Friday
# night — the night whose morning carries no obligation. "Free night" is therefore defined on the
# report date, not on the night's own date, and this note is quoted wherever the variable is used.
FREE_NIGHT_NOTE = ("A report dated D describes the night ending on the morning of D; a night is "
                   "'free' when D is a non-working day, so the sleeper had no alarm.")


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> pd.Timestamp:
    """`n`-th `weekday` (Mon=0) of a month; n=-1 gives the last."""
    days = pd.date_range(f"{year}-{month:02d}-01", periods=31, freq="D")
    days = days[days.month == month]
    hits = days[days.dayofweek == weekday]
    return hits[n if n >= 0 else len(hits) + n]


def holidays(years, region: str) -> set:
    """Fixed and rule-based public holidays for the two regions the cohorts proxy.

    The platform records language, not country, so this is a proxy: the English cohort is treated
    as anglophone-Northern-Hemisphere and the Russian cohort as following the Russian calendar.
    Mis-assignment attenuates every contrast built on these dates toward zero, which makes a null
    here weak evidence and a positive result conservative. The set is deliberately small and
    high-salience rather than exhaustive.
    """
    out = set()
    for y in years:
        if region == "en":
            out |= {pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y}-07-04"),
                    pd.Timestamp(f"{y}-12-24"), pd.Timestamp(f"{y}-12-25"),
                    pd.Timestamp(f"{y}-12-26"), pd.Timestamp(f"{y}-12-31"),
                    _nth_weekday(y, 11, 3, 3),            # US Thanksgiving (4th Thursday)
                    _nth_weekday(y, 9, 0, 0),             # US Labor Day (1st Monday)
                    _nth_weekday(y, 5, 0, -1)}            # US Memorial Day (last Monday)
        else:
            out |= {pd.Timestamp(f"{y}-{1:02d}-{d:02d}") for d in range(1, 9)}
            out |= {pd.Timestamp(f"{y}-02-23"), pd.Timestamp(f"{y}-03-08"),
                    pd.Timestamp(f"{y}-05-01"), pd.Timestamp(f"{y}-05-09"),
                    pd.Timestamp(f"{y}-06-12"), pd.Timestamp(f"{y}-11-04")}
    return out


def dst_transitions(years, region: str):
    """(spring_forward, fall_back) dates. Russia has observed no transitions since 2014."""
    if region != "en":
        return [], []
    spring = [_nth_weekday(y, 3, 6, 1) for y in years]     # US: 2nd Sunday in March
    fall = [_nth_weekday(y, 11, 6, 0) for y in years]      # US: 1st Sunday in November
    return spring, fall


def social_design(days, region: str) -> pd.DataFrame:
    """Calendar regressors on the report date, plus the seasonal and trend nuisance block.

    Columns fall into three blocks that the caller tests separately: `dow*` (the weekday profile),
    the social-time indicators (`free_night`, `holiday`, `holiday_eve`, `dst_spring`, `dst_fall`,
    `summer`), and the nuisance block (`trend`, two annual harmonics) that every arm carries.
    """
    idx = pd.DatetimeIndex(sorted(pd.DatetimeIndex(days).unique()))
    years = sorted({*idx.year, *(idx.year + 1)})
    hol = holidays(years, region)
    spring, fall = dst_transitions(years, region)
    dow = idx.dayofweek.to_numpy()
    doy = idx.dayofyear.to_numpy(float)
    t = (idx - idx.min()).days.to_numpy(float)

    X = pd.DataFrame(index=idx)
    X["trend"] = (t - t.mean()) / (t.std() + 1e-9)
    for k in (1, 2):
        X[f"s{k}"] = np.sin(2 * np.pi * k * doy / 365.25)
        X[f"c{k}"] = np.cos(2 * np.pi * k * doy / 365.25)
    for k in range(1, 7):
        X[f"dow{k}"] = (dow == k).astype(float)

    # The weekend is already carried by the weekday dummies, so a separate free-night indicator
    # would be collinear with them and identified only off the weekday holidays anyway. `holiday`
    # is therefore the free-night variable of record: adjusted for the weekday profile, its
    # coefficient contrasts a Tuesday that is a public holiday against an ordinary Tuesday, which
    # is the cleaner natural experiment. The weekend itself is read off the weekday contrast.
    is_hol = np.array([d in hol for d in idx], float)
    X["holiday"] = is_hol
    nxt = idx + pd.Timedelta(days=1)
    X["holiday_eve"] = np.array([(d in hol) for d in nxt], float) * (1 - is_hol)
    # A transition is scored on the mornings that follow it: the week after the spring shift is the
    # canonical acute social-jetlag window, and the autumn shift is its signed control.
    for name, dates in (("dst_spring", spring), ("dst_fall", fall)):
        v = np.zeros(len(idx))
        for d in dates:
            v[(idx >= d) & (idx < d + pd.Timedelta(days=7))] = 1.0
        X[name] = v
    X["summer"] = ((idx.month >= 7) & (idx.month <= 8)).astype(float)
    return X


NUISANCE = ["trend", "s1", "c1", "s2", "c2"]
WEEKDAY = [f"dow{k}" for k in range(1, 7)]
SOCIAL = ["holiday", "holiday_eve", "dst_spring", "dst_fall", "summer"]
