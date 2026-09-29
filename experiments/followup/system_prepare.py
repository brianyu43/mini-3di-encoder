"""Select real, nonduplicated throughput inputs from the already-local archive."""

import hashlib
import tarfile
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np

from experiments.common import ROOT, bounded_cpu, read, save, sha
from experiments.prepare import ARCHIVE
from mini3di_encoder.atoms import read_backbone
from mini3di_encoder.encode import encode_chain
from mini3di_encoder.learned import load_learned_model

from .common import OUT
from .evaluate_curve import encoded_arrays


def order(sid):
    return hashlib.sha256(("R3:" + sid).encode()).hexdigest()


def main():
    out = OUT / "R3"
    out.mkdir(exist_ok=False)
    bounded_cpu(out)
    protocol_path = Path(__file__).with_name("R3_PREREGISTRATION.json")
    protocol = read(protocol_path)
    manifest = read(OUT / "manifest-v2.json")
    queries = sorted(
        [r for r in manifest if r["split"] == "validation" and r["role"] == "query"],
        key=lambda r: order(r["record_id"]),
    )[:10]
    excluded_pdbs = {r["pdb"] for r in queries}
    candidates, failures = [], Counter()
    temp = out / "catalog-input.pdb"
    started = perf_counter()
    with tarfile.open(ARCHIVE, "r|gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            sid = Path(member.name).name
            if sid[1:5].lower() in excluded_pdbs:
                failures["query_pdb"] += 1
                continue
            data = archive.extractfile(member).read()
            chains = {line[21] for line in data.decode().splitlines() if line.startswith("ATOM  ")}
            if len(chains) != 1:
                failures["chain_count"] += 1
                continue
            temp.write_bytes(data)
            try:
                chain = read_backbone(temp, next(iter(chains)))
            except ValueError:
                failures["unsupported_pdb"] += 1
                continue
            length = len(chain.residues)
            if not 60 <= length <= 500:
                failures["length"] += 1
                continue
            if not chain.backbone_valid.all() or np.mean(chain.peptide_links()) < 0.95:
                failures["backbone_or_links"] += 1
                continue
            candidates.append(
                {
                    "record_id": sid,
                    "pdb": sid[1:5].lower(),
                    "aa": chain.aa,
                    "chain": chain.chain_id,
                    "length": length,
                    "aa_sha256": hashlib.sha256(chain.aa.encode()).hexdigest(),
                    "structure_sha256": hashlib.sha256(data).hexdigest(),
                    "archive_member": member.name,
                }
            )
    temp.unlink()
    accepted = []
    seen_aa, seen_hash = {r["aa_sha256"] for r in queries}, set()
    for r in sorted(candidates, key=lambda r: order(r["record_id"])):
        if r["aa_sha256"] in seen_aa or r["structure_sha256"] in seen_hash:
            failures["duplicate_aa_or_structure"] += 1
            continue
        seen_aa.add(r["aa_sha256"])
        seen_hash.add(r["structure_sha256"])
        accepted.append(r)
    conditions, skipped = [], []
    for low, high in [*protocol["target_length_strata"], [60, 500]]:
        eligible = [r for r in accepted if low <= r["length"] <= high]
        for count in protocol["target_sizes"]:
            if low == 60 and high == 500 and count != 5000:
                continue
            if len(eligible) < count:
                skipped.append(
                    {
                        "length": [low, high],
                        "target_count": count,
                        "available": len(eligible),
                        "reason": "insufficient distinct data",
                    }
                )
            else:
                conditions.append(
                    {
                        "name": f"length-{low}-{high}-n{count}",
                        "length": [low, high],
                        "target_count": count,
                        "target_ids": [r["record_id"] for r in eligible[:count]],
                    }
                )
    ids = {sid for c in conditions for sid in c["target_ids"]}
    selected = [r for r in accepted if r["record_id"] in ids]
    structures = out / "structures"
    structures.mkdir()
    needed = {r["archive_member"]: r for r in selected}
    with tarfile.open(ARCHIVE, "r|gz") as archive:
        for member in archive:
            if member.name in needed:
                row = needed.pop(member.name)
                content = archive.extractfile(member).read()
                assert hashlib.sha256(content).hexdigest() == row["structure_sha256"]
                (structures / f"{row['record_id']}.pdb").write_bytes(content)
    assert not needed
    for r in queries:
        (structures / f"{r['record_id']}.pdb").write_bytes((OUT / r["path"]).read_bytes())
    freeze = {
        "protocol": protocol,
        "protocol_sha256": sha(protocol_path),
        "archive_sha256": sha(ARCHIVE),
        "script_sha256": sha(__file__),
        "encoder_sha256": sha(ROOT / protocol["model"]),
        "matrix_sha256": sha(ROOT / protocol["matrix"]),
        "queries": queries,
        "targets": selected,
        "conditions": conditions,
        "skipped": skipped,
        "candidate_exclusions": dict(failures),
        "eligible_unique": len(accepted),
        "selection_seconds": perf_counter() - started,
        "no_search_results_used": True,
    }
    save(out / "data-freeze.json", freeze)
    model = load_learned_model(ROOT / protocol["model"])
    cache = out / "encoded"
    cache.mkdir()
    began = perf_counter()
    for i, r in enumerate(queries + selected):
        sid = r["record_id"]
        enc = encode_chain(read_backbone(structures / f"{sid}.pdb", r["chain"]), sid)
        features = {
            sid: {
                "x": np.array([v["features"] if v["valid"] else [0.0] * 10 for v in enc["rows"]]),
                "mask": np.array(enc["valid_seed_mask"]),
            }
        }
        arrays = encoded_arrays(*model, features)[sid]
        np.savez_compressed(cache / f"{sid}.npz", **arrays)
        if i % 500 == 0:
            print("R3 encoded", i, "of", len(queries) + len(selected), flush=True)
    save(
        out / "encoding.json",
        {
            "structures": len(queries) + len(selected),
            "seconds": perf_counter() - began,
            "encoded_files": {p.name: sha(p) for p in cache.iterdir()},
            "script_sha256": sha(__file__),
        },
    )
    print("R3 preparation complete", len(conditions), "conditions; skipped", skipped, flush=True)


if __name__ == "__main__":
    main()
