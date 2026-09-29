"""Restore exactly the frozen inputs into a new directory; never download a structure DB."""

import argparse
import shutil
import tarfile
import urllib.request
from pathlib import Path

from .common import DEFAULT_OUT, ROOT, command, environment, read, save, sha
from .prepare import ARCHIVE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--source-cache", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--allow-small-downloads",
        action="store_true",
        help="Allow pinned sources only, under 1 MiB total",
    )
    args = parser.parse_args()
    out = args.out.resolve()
    lock = read(ROOT / "experiments/source-lock.json")
    if not args.archive.is_file() or sha(args.archive) != lock["archive_sha256"]:
        raise ValueError("Supply the original scop40pdb.tar.gz with source-lock SHA-256")
    out.mkdir(parents=True, exist_ok=False)
    for name in [
        "manifest.json",
        "pair_plan.json",
        "data-freeze.json",
        "split_audit.json",
        "plan-at-freeze.md",
    ]:
        shutil.copyfile(ROOT / "experiments/frozen" / name, out / name)
    shutil.copyfile(ROOT / "experiments/config.json", out / "experiment_config.json")
    records = read(out / "manifest.json")
    needed = {r["archive_member"]: r for r in records}
    count = 0
    (out / "structures").mkdir()
    with tarfile.open(args.archive, "r|gz") as archive:
        for member in archive:
            if member.name not in needed:
                continue
            if not member.isfile():
                raise ValueError("Expected a regular PDB member")
            row = needed.pop(member.name)
            path = out / row["path"]
            path.write_bytes(archive.extractfile(member).read())
            assert sha(path) == row["structure_sha256"]
            count += 1
    assert count == len(records) and not needed
    assert sum(r["bytes"] for r in lock["files"]) < 1024 * 1024
    downloaded = 0
    for item in lock["files"]:
        target, cached = out / item["path"], args.source_cache / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if cached.is_file() and sha(cached) == item["sha256"]:
            shutil.copyfile(cached, target)
        elif args.allow_small_downloads:
            with urllib.request.urlopen(item["url"], timeout=30) as response:
                content = response.read(item["bytes"] + 1)
            if len(content) != item["bytes"]:
                raise ValueError("Unexpected download size")
            downloaded += len(content)
            target.write_bytes(content)
        else:
            raise FileNotFoundError(
                f"Missing pinned source: {cached}; allow small downloads explicitly"
            )
        assert sha(target) == item["sha256"]
    tool = out / "tools/usalign"
    command(
        ["clang++", "-O3", "-std=c++17", "TMalign.cpp", "-o", "TMalign"],
        tool / "compile.json",
        timeout=120,
        cwd=tool,
    )
    save(out / "environment.json", environment())
    save(
        out / "restore.json",
        {
            "structures": count,
            "source_lock_sha256": sha(ROOT / "experiments/source-lock.json"),
            "downloaded_bytes": downloaded,
            "tmalign_sha256": sha(tool / "TMalign"),
            "script_sha256": sha(Path(__file__)),
        },
    )
    print("Restored", count, "structures into", out, flush=True)


if __name__ == "__main__":
    main()
