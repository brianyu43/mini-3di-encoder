"""Provenance and bounded CPU helpers shared by the research commands."""

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "artifacts/sessions-11-15"
ARCHIVE_ROOT = ROOT.parent / "mini-3di-encoder-archive/session-01-05-60de6e8"
SEARCH_ROOT = Path(os.environ.get("MINI3DI_SEARCH_ROOT", ROOT.parent / "mini-3di-search"))
ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def bounded_cpu(out):
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["NUMBA_CACHE_DIR"] = str(out / "numba-cache")
    os.environ["MPLCONFIGDIR"] = str(out / "matplotlib-cache")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def order_key(value):
    return digest_bytes(("mini3di-20260929-v1:" + value).encode())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read(path):
    return json.loads(Path(path).read_text())


def command(argv, log, *, timeout=120, cwd=None):
    argv = [str(x) for x in argv]
    began = perf_counter()
    process = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, cwd=cwd)
    save(
        log,
        {
            "argv": argv,
            "returncode": process.returncode,
            "seconds": perf_counter() - began,
            "stdout": process.stdout,
            "stderr": process.stderr,
        },
    )
    process.check_returncode()
    return process.stdout


def environment():
    import importlib.metadata

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "torch", "matplotlib", "psutil", "numba", "biopython"]
        },
        "encoder_commit": subprocess.check_output(
            ["git", "-C", ROOT, "rev-parse", "HEAD"], text=True
        ).strip(),
    }
