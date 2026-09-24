"""Fetch the planetary geomagnetic Kp/ap index (GFZ Potsdam) -> external/signals + data-card.

Kp (0-9, 3-hourly) measures global geomagnetic disturbance driven by solar activity. It is the
predictor for the "geomagnetic storms disturb our dreams" myth (docs/BOLD-QUESTIONS.md Q3):
there is a real (contested) literature linking Kp to sleep, mood and cardiac events, so a rigorous
test — positive OR null — is worthwhile.

We derive per-day: kp_max (peak of the 8 3-hourly values = worst disturbance that day), kp_mean,
Ap (daily equivalent amplitude), and storm flags (kp_max>=5 => G1+ storm). Public data, CC BY 4.0.

    python -m psychohistory.signals.spaceweather
"""
from __future__ import annotations

import io

import pandas as pd
import requests

from .. import config as C
from ..utils.io import write_datacard

GFZ_URL = "https://kp.gfz-potsdam.de/app/files/Kp_ap_Ap_SN_F107_since_1932.txt"
# fixed column layout (see file header): YYY MM DD days days_m Bsr dB Kp1..8 ap1..8 Ap SN F10.7o F10.7a D
_KP_COLS = slice(7, 15)
_AP_DAILY = 23


def fetch_kp(url: str = GFZ_URL, timeout: int = 120) -> pd.DataFrame:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    recs = []
    for line in io.StringIO(r.text):
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        y, m, d = int(p[0]), int(p[1]), int(p[2])
        kp = [float(x) for x in p[_KP_COLS]]
        recs.append((f"{y:04d}-{m:02d}-{d:02d}", max(kp), sum(kp) / len(kp), float(p[_AP_DAILY])))
    df = pd.DataFrame(recs, columns=["date", "kp_max", "kp_mean", "ap_daily"])
    df["date"] = pd.to_datetime(df["date"])
    df["storm_g1"] = (df.kp_max >= 5.0).astype(int)      # G1+ geomagnetic storm
    df["storm_strong"] = (df.kp_max >= 6.0).astype(int)  # G2+ (stronger)
    return df


def main():
    out = C.EXTERNAL / "signals"
    out.mkdir(parents=True, exist_ok=True)
    df = fetch_kp()
    fp = out / "kp_daily.csv"
    df.to_csv(fp, index=False)
    write_datacard(
        C.MANIFESTS, "kp_daily", source=GFZ_URL,
        license="CC BY 4.0 (GFZ Potsdam; Matzka et al. 2021)", sensitivity="S0",
        local_path=str(fp), notes="Daily geomagnetic Kp (max/mean of 8 3-hourly), Ap, storm flags.",
        extra={"rows": len(df), "date_min": str(df.date.min().date()),
               "date_max": str(df.date.max().date())})
    print(f"[kp] {len(df)} days -> {fp}  ({df.date.min().date()}..{df.date.max().date()})")
    print(df.tail(3).to_string())


if __name__ == "__main__":
    main()
