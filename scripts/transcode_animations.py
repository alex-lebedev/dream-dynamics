"""Transcode animated GIFs under `60-results/` into web-playable H.264 and VP9.

GIF is the wrong delivery format for a supplementary animation: it is an order of magnitude larger
than H.264 at the same visual quality, caps at 256 colours, and cannot be scrubbed. Analyses that
already emit a GIF (matplotlib's `PillowWriter` path) keep doing so; this turns each one into an
`.mp4` and a `.webm` beside it without re-running the analysis.

New animations should call `psychohistory.utils.anim.render`, which writes video directly. This
script exists for the ones that predate it.

    python scripts/transcode_animations.py            # all GIFs under 60-results/
    python scripts/transcode_animations.py --dry-run
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "60-results"

# GIF frame delays are quantised in hundredths of a second and are often irregular; forcing a
# constant output frame rate avoids the stutter that a straight copy produces. yuv420p plus
# +faststart is the combination browsers will actually decode.
H264 = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "slow",
        "-profile:v", "high", "-level", "4.0", "-movflags", "+faststart"]
VP9 = ["-c:v", "libvpx-vp9", "-pix_fmt", "yuv420p", "-crf", "32", "-b:v", "0", "-row-mt", "1"]
# H.264 in yuv420p needs even dimensions; GIFs frequently are not.
EVEN = "scale=trunc(iw/2)*2:trunc(ih/2)*2"


def transcode(gif: Path, fps: int, dry: bool) -> None:
    for ext, args in (("mp4", H264), ("webm", VP9)):
        out = gif.with_suffix("." + ext)
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(gif),
               "-vf", f"fps={fps},{EVEN}", *args, str(out)]
        if dry:
            print("  would run:", " ".join(cmd))
            continue
        subprocess.run(cmd, check=True)
        print(f"  {out.relative_to(ROOT)}  "
              f"{gif.stat().st_size / 1e6:.1f} MB -> {out.stat().st_size / 1e6:.1f} MB")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fps", type=int, default=12, help="constant output frame rate (default 12)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if not shutil.which("ffmpeg"):
        print("error: ffmpeg not found on PATH", file=sys.stderr)
        return 1

    gifs = sorted(RESULTS.rglob("*.gif"))
    if not gifs:
        print("no GIFs found under 60-results/")
        return 0
    for gif in gifs:
        print(gif.relative_to(ROOT))
        transcode(gif, a.fps, a.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
