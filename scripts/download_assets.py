"""Fetch the third-party assets the project needs but does not vendor.

* **Stockfish** (GPL-3.0) -- the classical engine used for human-side analysis
  and as a strength yardstick for the neural network.
* **cburnett piece set** (CC BY-SA 3.0, from the Lichess repository) -- the
  board graphics. Note: chess.com's own piece set is proprietary and is
  deliberately *not* used here.

Usage::

    python scripts/download_assets.py             # everything
    python scripts/download_assets.py --pieces    # just the SVGs
    python scripts/download_assets.py --stockfish
"""

from __future__ import annotations

import argparse
import io
import platform
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENGINES_DIR = PROJECT_ROOT / "engines"
PIECES_DIR = PROJECT_ROOT / "frontend" / "public" / "pieces"

USER_AGENT = "AI_Chess_Claude/0.1 (asset downloader)"

PIECE_SETS = {
    "cburnett": "https://raw.githubusercontent.com/lichess-org/lila/master/public/piece/cburnett",
}
PIECE_CODES = [f"{colour}{piece}" for colour in "wb" for piece in "PNBRQK"]

STOCKFISH_RELEASE = "https://api.github.com/repos/official-stockfish/Stockfish/releases/latest"
STOCKFISH_ASSETS = {
    ("Windows", "AMD64"): "stockfish-windows-x86-64-avx2.zip",
    ("Windows", "ARM64"): "stockfish-windows-armv8.zip",
    ("Linux", "x86_64"): "stockfish-ubuntu-x86-64-avx2.tar",
    ("Darwin", "arm64"): "stockfish-macos-m1-apple-silicon.tar",
    ("Darwin", "x86_64"): "stockfish-macos-x86-64-avx2.tar",
}


def _get(url: str, timeout: int = 120) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


# --------------------------------------------------------------------------- #
# Pieces
# --------------------------------------------------------------------------- #


def download_pieces(sets=("cburnett",)) -> int:
    written = 0
    for name in sets:
        base = PIECE_SETS[name]
        target = PIECES_DIR / name
        target.mkdir(parents=True, exist_ok=True)
        for code in PIECE_CODES:
            destination = target / f"{code}.svg"
            if destination.exists() and destination.stat().st_size > 100:
                continue
            try:
                destination.write_bytes(_get(f"{base}/{code}.svg", timeout=30))
                written += 1
                print(f"  + {name}/{code}.svg")
            except urllib.error.URLError as error:
                print(f"  ! {name}/{code}.svg failed: {error}", file=sys.stderr)
    _write_attribution()
    return written


def _write_attribution() -> None:
    PIECES_DIR.mkdir(parents=True, exist_ok=True)
    (PIECES_DIR / "ATTRIBUTION.md").write_text(
        "# Piece artwork\n\n"
        "The `cburnett` set is by Colin M.L. Burnett, distributed with Lichess\n"
        "(https://github.com/lichess-org/lila/tree/master/public/piece) under\n"
        "**CC BY-SA 3.0**. Redistribution must keep the attribution and licence.\n\n"
        "chess.com's piece artwork is proprietary and is intentionally not used\n"
        "by this project.\n",
        encoding="utf-8",
    )


# --------------------------------------------------------------------------- #
# Stockfish
# --------------------------------------------------------------------------- #


def download_stockfish() -> Path | None:
    import json

    key = (platform.system(), platform.machine())
    asset_name = STOCKFISH_ASSETS.get(key)
    if asset_name is None:
        print(f"no Stockfish asset mapped for {key}; install it manually", file=sys.stderr)
        return None

    existing = _find_existing_stockfish()
    if existing:
        print(f"  = Stockfish already present: {existing}")
        return existing

    print("  . querying latest Stockfish release")
    release = json.loads(_get(STOCKFISH_RELEASE, timeout=60))
    match = next((a for a in release["assets"] if a["name"] == asset_name), None)
    if match is None:
        print(f"asset {asset_name} not in release {release.get('tag_name')}", file=sys.stderr)
        return None

    print(f"  . downloading {asset_name} ({match['size'] // (1024 * 1024)} MB)")
    payload = _get(match["browser_download_url"], timeout=600)
    ENGINES_DIR.mkdir(parents=True, exist_ok=True)

    if asset_name.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            archive.extractall(ENGINES_DIR)
    else:
        import tarfile

        with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
            archive.extractall(ENGINES_DIR)

    binary = _find_existing_stockfish()
    if binary:
        if platform.system() != "Windows":
            binary.chmod(0o755)
        print(f"  + Stockfish at {binary}")
    return binary


def _find_existing_stockfish() -> Path | None:
    suffix = ".exe" if platform.system() == "Windows" else ""
    if not ENGINES_DIR.exists():
        return None
    for candidate in sorted(ENGINES_DIR.rglob(f"stockfish*{suffix}")):
        if candidate.is_file() and (suffix or candidate.stat().st_mode & 0o111 or True):
            return candidate
    return shutil.which("stockfish") and Path(shutil.which("stockfish"))  # type: ignore[return-value]


# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pieces", action="store_true")
    parser.add_argument("--stockfish", action="store_true")
    args = parser.parse_args()
    everything = not (args.pieces or args.stockfish)

    if everything or args.pieces:
        print("pieces:")
        count = download_pieces()
        print(f"  {count} file(s) downloaded")

    if everything or args.stockfish:
        print("stockfish:")
        download_stockfish()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
