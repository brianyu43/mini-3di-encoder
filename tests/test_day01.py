from pathlib import Path

import pytest

from mini3di_encoder.day01 import attach_3di_headers, one_fasta, run_day01


def test_multi_chain_reference_is_rejected(tmp_path):
    path = tmp_path / "chains.fasta"
    path.write_text(">A\nAA\n>B\nAA\n")
    with pytest.raises(ValueError, match="exactly one"):
        one_fasta(path)


def test_existing_output_is_preserved(tmp_path):
    sentinel = tmp_path / "do-not-change.txt"
    sentinel.write_text("original")
    with pytest.raises(FileExistsError):
        run_day01(Path("does-not-exist.pdb"), "A", Path("no-binary"), tmp_path)
    assert sentinel.read_text() == "original"


def test_header_identity_must_match_before_copy(tmp_path):
    for name, key in [("db", 0), ("db_ss", 1), ("db_h", 0)]:
        (tmp_path / (name + ".index")).write_text(f"{key}\t0\t4\n")
    with pytest.raises(ValueError, match="keys differ"):
        attach_3di_headers(tmp_path / "db")
    assert not (tmp_path / "db_ss_h").exists()


def test_verified_headers_are_preserved_byte_exact(tmp_path):
    for name in ["db", "db_ss", "db_h"]:
        (tmp_path / (name + ".index")).write_text("7\t0\t4\n")
    (tmp_path / "db_h").write_bytes(b"A\n\0")
    (tmp_path / "db_h.dbtype").write_bytes(b"\x0c\0\0\0")
    attach_3di_headers(tmp_path / "db")
    for suffix in ["", ".index", ".dbtype"]:
        assert (tmp_path / ("db_ss_h" + suffix)).read_bytes() == (
            tmp_path / ("db_h" + suffix)
        ).read_bytes()
