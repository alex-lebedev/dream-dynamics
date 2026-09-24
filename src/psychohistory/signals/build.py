"""Fetch daily societal signals into 10-data/external/signals/ and write data-cards.

No-key sources implemented here: FRED (VIX, S&P 500) and Geotone/GDELT daily news tone.
Keyed sources (ACLED) and pytrends live in scripts/ + RUNBOOK.md. All signals are public
(S0) except where a provider's terms say otherwise.

Usage:
    python -m psychohistory.signals.build --fred VIXCLS SP500 --geotone
"""
from __future__ import annotations
import argparse
import io
from pathlib import Path

import pandas as pd
import requests

from .. import config as C
from ..utils.io import write_datacard

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
GEOTONE = "https://huggingface.co/datasets/davidcummings/geotone-global-news-signals/resolve/main/{f}"


def fetch_fred(series_id: str, timeout: int = 60) -> pd.DataFrame:
    r = requests.get(FRED_CSV.format(sid=series_id), timeout=timeout)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"].replace(".", pd.NA), errors="coerce")
    return df.dropna().reset_index(drop=True)


def fetch_geotone(fname: str = "topic_daily.csv", timeout: int = 120) -> pd.DataFrame:
    r = requests.get(GEOTONE.format(f=fname), timeout=timeout)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fred", nargs="*", default=[])
    ap.add_argument("--geotone", action="store_true")
    ap.add_argument("--outdir", default=str(C.EXTERNAL / "signals"))
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    for sid in args.fred:
        df = fetch_fred(sid)
        fp = out / f"fred_{sid}.csv"
        df.to_csv(fp, index=False)
        write_datacard(C.MANIFESTS, f"fred_{sid}", source=FRED_CSV.format(sid=sid),
                       license="public (FRED/St. Louis Fed)", sensitivity="S0",
                       local_path=str(fp), notes=f"FRED series {sid}",
                       extra={"rows": len(df), "date_min": str(df.date.min().date()),
                              "date_max": str(df.date.max().date())})
        print(f"[fred] {sid}: {len(df)} rows -> {fp}")

    if args.geotone:
        for f in ["topic_daily.csv", "country_daily.csv"]:
            df = fetch_geotone(f)
            fp = out / f"geotone_{f}"
            df.to_csv(fp, index=False)
            write_datacard(C.MANIFESTS, f"geotone_{f.split('.')[0]}", source=GEOTONE.format(f=f),
                           license="CC BY 4.0 (Geotone/GDELT)", sensitivity="S0",
                           local_path=str(fp), notes="GDELT-derived daily news tone/attention",
                           extra={"rows": len(df)})
            print(f"[geotone] {f}: {len(df)} rows -> {fp}")


if __name__ == "__main__":
    main()
