"""Reproducible video rendering for `psychohistory` figures.

Companion to `plotstyle.py`: same Okabe-Ito palette and clean-spine look, but tuned for motion and
for playback in a browser or a journal's supplementary viewer.

    from psychohistory.utils.anim import apply_video_style, render, PALETTE
    apply_video_style()
    fig, ax = plt.subplots(figsize=(12.8, 7.2))   # 1280x720 at dpi=100
    ...
    render(fig, update, n_frames=300, out_stem=OUT / "arrow", fps=30)

Writes `arrow.mp4` (H.264, web-safe) and `arrow.webm` (VP9), and optionally `arrow.gif`.

Two constraints motivate the differences from `plotstyle`:

- **Frames must all be the same size**, so `savefig.bbox` cannot be `"tight"`. Lay figures out with
  `constrained_layout` or explicit margins instead.
- **H.264 in `yuv420p` requires even pixel dimensions**, and `yuv420p` is what browsers and
  PowerPoint will actually decode. `render` checks the geometry and fails loudly rather than emitting
  a file that silently will not play.

Fonts are set larger than print: a supplementary video is read at a quarter of the screen.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

from psychohistory.utils.plotstyle import CYCLE, PALETTE  # noqa: F401  (re-exported)

__all__ = ["apply_video_style", "render", "PALETTE", "CYCLE", "have_ffmpeg"]


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def apply_video_style() -> None:
    """`plotstyle.apply_style()` adapted for animation: fixed frame size, larger type."""
    mpl.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": 100,
        # Must NOT be "tight" — a variable bounding box changes the frame size mid-render.
        "savefig.bbox": None,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.labelsize": 13,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.7,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "legend.frameon": False,
        "lines.linewidth": 2.2,
        "axes.prop_cycle": mpl.cycler(color=CYCLE),
    })


def _encode(src: Path, dst: Path, args: list[str]) -> None:
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), *args, str(dst)]
    subprocess.run(cmd, check=True)


def render(fig, update, n_frames: int, out_stem, fps: int = 30,
           formats: tuple[str, ...] = ("mp4", "webm"), bitrate: int = 6000,
           gif_stride: int = 3) -> list[str]:
    """Render `update(frame_index)` over `n_frames` into web-playable video.

    `update` is called with the frame index and should mutate the figure in place. Returns the paths
    written. `gif_stride` subsamples frames for the (much larger, lower quality) GIF fallback, which
    is only produced if "gif" is in `formats`.
    """
    stem = Path(out_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)

    w, h = (int(round(d * fig.dpi)) for d in fig.get_size_inches())
    if w % 2 or h % 2:
        raise ValueError(
            f"frame is {w}x{h}px; H.264 yuv420p needs even dimensions. Adjust figsize/dpi "
            f"(e.g. figsize=(12.8, 7.2) at dpi=100 -> 1280x720)."
        )

    written: list[str] = []
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=False)

    if {"mp4", "webm"} & set(formats):
        if not have_ffmpeg():
            raise RuntimeError("ffmpeg not found; install it or pass formats=('gif',)")
        master = stem.with_suffix(".mp4")
        # -pix_fmt yuv420p and +faststart are what make this play in a browser rather than
        # only in VLC; without them the file is technically valid and practically useless.
        writer = FFMpegWriter(
            fps=fps, bitrate=bitrate, codec="libx264",
            extra_args=["-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "4.0",
                        "-movflags", "+faststart"],
        )
        anim.save(str(master), writer=writer, dpi=fig.dpi)
        written.append(str(master))
        if "webm" in formats:
            webm = stem.with_suffix(".webm")
            _encode(master, webm, ["-c:v", "libvpx-vp9", "-b:v", f"{bitrate}k",
                                   "-pix_fmt", "yuv420p", "-row-mt", "1"])
            written.append(str(webm))
        if "mp4" not in formats:
            master.unlink()
            written.remove(str(master))

    if "gif" in formats:
        gif = stem.with_suffix(".gif")
        anim.save(str(gif), writer=PillowWriter(fps=max(1, fps // gif_stride)), dpi=fig.dpi)
        written.append(str(gif))

    plt.close(fig)
    for p in written:
        print(f"[anim] wrote {p} ({Path(p).stat().st_size / 1e6:.1f} MB)")
    return written
