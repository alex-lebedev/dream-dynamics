"""Give the nightmare branching ratio the null it never had.

`2026-07-19-13-nightmare-contagion.py` reports a post-spike branching ratio of 1.15 for nightmares
against 0.97 for a food-theme control, and writes `perm_p=np.nan` for both: there is no null
distribution, no confidence interval and no p-value anywhere in that table. Section 3.5 nonetheless
read 1.15 and 0.97 as "both near unity" and concluded flatly that nightmares do not propagate
through the population. A ~15% excess over baseline is not self-evidently near unity, and a negative
control is a comparison rather than an inference — it tells you the statistic behaves differently on
a neutral theme, not whether either value is distinguishable from chance.

The statistic is reimplemented here rather than imported, for a reason that is about feasibility
rather than taste: the published `branching_ratio` re-derives a quantile inside a per-day Python loop
over a DataFrame, which costs seconds per call and makes a 1,000-draw null a multi-hour job. The
vectorized version below is asserted equal to the published value to four decimals before any null
is drawn, so the reimplementation is checked rather than trusted. One property of the published
statistic is worth naming because its docstring misdescribes it: the baseline is the event rate
following *all non-spike days*, not a weekday-matched set, so the ratio carries whatever weekday
structure differs between spike and non-spike days. That is precisely what the displacement nulls below
control for.

A CORRECTION TO THIS SCRIPT'S FIRST VERSION, which is why the arms are named as they are. That version
rotated the whole daily series, `np.roll(v, k*7)`, and described it as "breaking only the alignment
between spike days and the days that follow them". It does no such thing. Both the spike days and the
event days are derived from the same rotated vector, and the series is contiguous, so rolling it
translates every spike and its followers together and leaves the statistic nearly unchanged: over the
nineteen shortest rotations the nightmare ratio took eleven distinct values in 1.056-1.163 around an
observed 1.149, and the food control's rotated values all sat *above* its observed value. The reported
"null mean identical to the observed value" was the signature of a near-invariant transformation, and
the manuscript had read that degeneracy as its strongest evidence against propagation. A null must move
the thing being tested; this one is the cautionary example.

Three nulls, all of which displace or destroy the spike-to-follower alignment rather than preserving it:

  shuffle        the daily values are permuted across dates. The direct analogue of the calendar-shuffle
                 null already used for the g(r) pair correlation; destroys temporal and weekday
                 structure together, so it also destroys the persistence that could innocently explain
                 the excess. Loosest of the three.

  displace_week  the *event series stays fixed* and the spike day set is displaced circularly by a
                 random whole number of weeks. The series keeps its autocorrelation, and the spikes keep
                 their weekday phase and their count; only the correspondence between a spike and the
                 three days after it is randomized. This is the arm that isolates propagation. (One
                 wrapped spike per draw can land on the series' single missing day, so the effective
                 spike count is 85 or 86 rather than exactly 86 — immaterial at this ratio.) The
                 support is enumerated exhaustively rather than sampled, so its p-value is exact.

  displace_day   the same displacement at single-day resolution. It sacrifices the weekday phase that
                 `displace_week` protects, in exchange for a support of ~850 rather than ~120 distinct
                 draws. Reported because a p-value from `displace_week` cannot be finer than ~1/120,
                 and a reader is entitled to know when granularity is doing the work.

All three pass through the identical baseline construction as the observed statistic, so any tendency of
the procedure itself to manufacture a ratio above one is absorbed into the null rather than credited to
the data. Every arm reports `n_distinct_null` against the size of its support, so a degenerate null
announces itself in the output table instead of having to be caught in review.

    PYTHONPATH=src PYTHONNOUSERSITE=1 python3 analyses/2026-08-21-06-branching-null.py
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from psychohistory import config as C

SRC = Path("analyses/2026-07-19-13-nightmare-contagion.py")
TABLES = C.RESULTS / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

N_PERM = 1000
SEED = 0
LAGS = (1, 2, 3)


def _load_module():
    spec = importlib.util.spec_from_file_location("nightmare_contagion", SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rng(seed: int, *parts: str) -> np.random.Generator:
    """Deterministic per-cell seeding.

    Python's builtin `hash` is salted per interpreter process, so seeding from it makes every
    p-value here irreproducible across runs — a bug this project already found and fixed once in the
    surrogate code, and then reintroduced in the first version of this script. blake2b of the cell
    identity is stable across processes and machines.
    """
    key = "|".join((str(seed), *parts)).encode()
    return np.random.default_rng(int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big"))


def _grid(dates: pd.Series) -> tuple[np.ndarray, int]:
    d = pd.to_datetime(dates).dt.normalize()
    off = (d - d.min()).dt.days.values.astype(int)
    return off, int(off.max()) + 1


def branching_fast(values: np.ndarray, off: np.ndarray, span: int,
                   q_spike: float = 0.90, q_event: float = 0.75,
                   spike_off: np.ndarray | None = None) -> dict:
    """Vectorized equivalent of the published branching ratio.

    post  = P(day is a top-quartile event | it is 1-3 days after a top-decile spike day)
    base  = P(day is a top-quartile event | it is 1-3 days after a non-spike day)
    ratio = post / base

    `spike_off` decouples the two halves of the statistic. Left None it reproduces the published
    behaviour exactly: spikes are the top-decile days of `values`, and the events they are checked
    against are the top-quartile days of the same `values`. Passed an array of day offsets, the event
    series is still built from `values` but the spike days are taken to be those offsets — which is
    what makes the displacement null below able to move one relative to the other.
    """
    thr_spike = np.quantile(values, q_spike)
    thr_event = np.quantile(values, q_event)

    ev = np.zeros(span + max(LAGS) + 1, bool)
    ev[off] = values >= thr_event          # absent days stay False, as in the published version

    if spike_off is None:
        spike = values >= thr_spike
        src_post, src_base = off[spike], off[~spike]
        n_spikes = int(spike.sum())
    else:
        # A displaced spike set: the same number of spike days, relocated. Membership is resolved on
        # the offset grid rather than on the row index, so the wrapped days land on real calendar
        # positions and the missing day in the series is handled the same way it is for the observed
        # statistic (`ev` is False there).
        mark = np.zeros(span + max(LAGS) + 1, bool)
        mark[spike_off % span] = True
        is_spike = mark[off]
        src_post, src_base = off[is_spike], off[~is_spike]
        n_spikes = int(is_spike.sum())

    out = {}
    for name, src in (("post", src_post), ("base", src_base)):
        hits = np.concatenate([ev[src + lag] for lag in LAGS]) if len(src) else np.array([], bool)
        out[f"{name}_rate"] = float(hits.mean()) if hits.size else np.nan
    br = out["post_rate"] / out["base_rate"] if out["base_rate"] else np.nan
    return {**out, "branching_ratio": float(br), "n_spikes": n_spikes}


def main() -> None:
    mod = _load_module()
    daily = mod.load_daily()
    off, span = _grid(daily["date"])
    print(f"[data] {len(daily)} EN days {daily.date.min().date()}..{daily.date.max().date()}")

    # Verify the reimplementation against the published statistic before drawing any null.
    print("\n[verify] vectorized statistic vs published implementation")
    for col in ("nightmare_index", "food"):
        pub = mod.branching_ratio(daily, col, q=0.90)
        fast = branching_fast(daily[col].values, off, span)
        ok = abs(pub["branching_ratio"] - fast["branching_ratio"]) < 1e-4
        print(f"    {col:16s} published {pub['branching_ratio']:.4f} | "
              f"vectorized {fast['branching_ratio']:.4f} | match={ok}")
        assert ok, f"reimplementation diverges for {col}; do not trust the null"

    rows = []
    for col, label in (("nightmare_index", "nightmare"), ("food", "food (negative control)")):
        v = daily[col].values
        obs = branching_fast(v, off, span)
        br = obs["branching_ratio"]
        spike_off = off[v >= np.quantile(v, 0.90)]

        for kind in ("shuffle", "displace_week", "displace_day"):
            rng = _rng(SEED, col, kind)
            # A displacement null has a finite support, so enumerate it rather than sampling it: the
            # p-value becomes exact and `n_distinct_null` describes the whole null instead of a draw
            # from it. The shuffle arm has no such support and stays at N_PERM random draws.
            step = 7 if kind == "displace_week" else 1
            shifts = np.arange(step, span, step) if kind != "shuffle" else None
            n_draws = N_PERM if shifts is None else len(shifts)
            s = np.empty(n_draws)
            for i in range(n_draws):
                if kind == "shuffle":
                    s[i] = branching_fast(rng.permutation(v), off, span)["branching_ratio"]
                else:
                    shift = shifts[i]
                    s[i] = branching_fast(v, off, span,
                                          spike_off=spike_off + shift)["branching_ratio"]
            s = s[np.isfinite(s)]
            p = float((1 + np.sum(s >= br)) / (len(s) + 1))
            # How many distinct statistics the null can even produce. A displacement null on a series
            # of this length has a finite support, and a p-value cannot be finer than 1/|support|;
            # reporting it stops the reader from over-reading a precise-looking number.
            support = n_draws
            rows.append({
                "series": label, "null": kind, "n_spikes": obs["n_spikes"],
                "observed_br": round(br, 4),
                "post_rate": round(obs["post_rate"], 4),
                "base_rate": round(obs["base_rate"], 4),
                "null_mean": round(float(s.mean()), 4),
                "null_sd": round(float(s.std()), 4),
                "null_q95": round(float(np.quantile(s, 0.95)), 4),
                "n_distinct_null": int(len(np.unique(np.round(s, 6)))),
                "support": int(support),
                "n_null": int(len(s)), "p": round(p, 4),
            })
            print(f"    {label:26s} {kind:13s}: observed {br:.3f} | null mean {s.mean():.3f} "
                  f"(sd {s.std():.3f}) q95 {np.quantile(s, 0.95):.3f} | p={p:.3f} "
                  f"| {len(np.unique(np.round(s, 6)))} distinct of {support} possible")

    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "branching_ratio_null.csv", index=False)
    print(f"\n[write] {TABLES / 'branching_ratio_null.csv'}")
    print("\nRead: p>.05 means the observed ratio is not distinguishable from what the null "
          "manufactures, which is what licenses the 'no propagation' reading. p<.05 on any "
          "null would mean Section 3.5 has to be rewritten. Check `n_distinct_null` against "
          "`support` before believing any of it: a displacement null whose statistic barely moves "
          "is not testing anything, which is how the first version of this script went wrong.")


if __name__ == "__main__":
    main()
