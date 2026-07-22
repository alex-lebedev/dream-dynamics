"""Provenance helpers: checksums and data-cards."""
from __future__ import annotations
import hashlib
import json
from datetime import date
from pathlib import Path


def sha256(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def human_size(nbytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(nbytes) < 1024:
            return f"{nbytes:.1f}{unit}"
        nbytes /= 1024
    return f"{nbytes:.1f}PB"


def write_datacard(manifests_dir, name: str, source: str, license: str,
                   sensitivity: str, local_path=None, notes: str = "", extra: dict | None = None):
    """Write a machine-readable data-card JSON into 10-data/manifests/."""
    manifests_dir = Path(manifests_dir)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    card = {
        "name": name,
        "source": source,
        "license": license,
        "sensitivity": sensitivity,
        "retrieved": str(date.today()),
        "notes": notes,
    }
    if local_path and Path(local_path).exists():
        p = Path(local_path)
        card["local_path"] = str(p)
        card["bytes"] = p.stat().st_size
        card["size"] = human_size(p.stat().st_size)
        card["sha256"] = sha256(p)
    if extra:
        card.update(extra)
    out = manifests_dir / f"{name}.datacard.json"
    out.write_text(json.dumps(card, indent=2))
    return out
