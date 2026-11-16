"""Extract the user-supplied AAU archive to ignored local data storage."""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    arguments = parser.parse_args()
    if not arguments.archive.is_file():
        raise FileNotFoundError(arguments.archive)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(arguments.archive, "r:xz") as archive:
        members = [member for member in archive.getmembers() if member.isfile() and member.name.startswith("AAU/")]
        archive.extractall(arguments.output_dir, members=members, filter="data")
    print(f"Extracted {len(members)} files to {arguments.output_dir / 'AAU'}")


if __name__ == "__main__":
    main()

