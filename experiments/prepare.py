"""Select and freeze a fold-disjoint, previously unobserved small SCOPe experiment."""

import argparse
import itertools
import json
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from mini3di_encoder.atoms import read_backbone

from .common import DEFAULT_OUT, SEARCH_ROOT, digest_bytes, environment, order_key, read, save, sha

ARCHIVE = SEARCH_ROOT / "artifacts/m3-scop-archive-20260905T061830Z/downloads/scop40pdb.tar.gz"
LABELS = SEARCH_ROOT / (
    "artifacts/m3-downloads-20260905T060403Z/downloads/"
    "analysis__scopbenchmark__data__scop_lookup.fix.tsv"
)
PREVIOUS = [
    SEARCH_ROOT / "artifacts/m3-pilot-data-20260905T063500Z/manifest.jsonl",
    SEARCH_ROOT / "artifacts/m5-d2-20260906-v2/manifest.jsonl",
]


def previous_ids():
    ids = {"d1ubqa_"}
    for path in PREVIOUS:
        ids.update(json.loads(line)["record_id"] for line in path.read_text().splitlines())
    return ids


def catalog(out):
    labels = dict(line.split() for line in LABELS.read_text().splitlines())
    previous = previous_ids()
    pdb_exclude = {sid[1:5].lower() for sid in previous}
    accepted, excluded = [], Counter()
    temporary = out / "catalog-input.pdb"
    with tarfile.open(ARCHIVE, "r|gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            sid = Path(member.name).name
            label = labels.get(sid, "")
            if not label or label[0] not in "abcd":
                excluded["no_supported_class"] += 1
                continue
            if sid in previous or sid[1:5].lower() in pdb_exclude:
                excluded["previous_domain_or_pdb"] += 1
                continue
            data = archive.extractfile(member).read()
            lines = data.decode().splitlines()
            chains = {line[21] for line in lines if line.startswith("ATOM  ")}
            if len(chains) != 1:
                excluded["multiple_or_no_chains"] += 1
                continue
            temporary.write_bytes(data)
            try:
                chain = read_backbone(temporary, next(iter(chains)))
            except ValueError:
                excluded["unsupported_pdb"] += 1
                continue
            n = len(chain.residues)
            if not 60 <= n <= 250:
                excluded["length"] += 1
                continue
            if not chain.backbone_valid.all():
                excluded["missing_backbone"] += 1
                continue
            if np.mean(chain.peptide_links()) < 0.95:
                excluded["too_many_chain_breaks"] += 1
                continue
            ca = chain.xyz[:, 1]
            distances = np.linalg.norm(ca[:, None] - ca[None, :], axis=-1)
            fingerprint = np.rint(distances[np.tril_indices(n, -1)] * 100).astype("<i4")
            accepted.append(
                {
                    "record_id": sid,
                    "pdb": sid[1:5].lower(),
                    "family": label,
                    "superfamily": ".".join(label.split(".")[:3]),
                    "fold": ".".join(label.split(".")[:2]),
                    "class": label[0],
                    "chain": chain.chain_id,
                    "length": n,
                    "aa": chain.aa,
                    "aa_sha256": digest_bytes(chain.aa.encode()),
                    "shape_sha256": digest_bytes(fingerprint.tobytes()),
                    "structure_sha256": digest_bytes(data),
                    "archive_member": member.name,
                }
            )
    temporary.unlink()
    result = {
        "accepted": accepted,
        "exclusions": dict(excluded),
        "archive_sha256": sha(ARCHIVE),
        "labels_sha256": sha(LABELS),
        "previous_manifest_sha256": {str(p): sha(p) for p in PREVIOUS},
        "previous_pdb_count": len(pdb_exclude),
    }
    save(out / "catalog.json", result)
    return result


def select(records, quotas):
    groups = defaultdict(list)
    for row in records:
        groups[row["superfamily"]].append(row)
    used = {key: set() for key in ("fold", "pdb", "aa_sha256", "shape_sha256")}
    selected = []
    for split, quota in quotas.items():
        for cls in itertools.islice(itertools.cycle("abcd"), quota):
            candidates = sorted(
                (sf for sf, rows in groups.items() if rows[0]["class"] == cls), key=order_key
            )
            chosen = None
            for sf in candidates:
                if groups[sf][0]["fold"] in used["fold"]:
                    continue
                batch = []
                local = {k: used[k].copy() for k in ("pdb", "aa_sha256", "shape_sha256")}
                for row in sorted(groups[sf], key=lambda r: order_key(r["record_id"])):
                    if any(row[k] in local[k] for k in local):
                        continue
                    batch.append(row)
                    for k in local:
                        local[k].add(row[k])
                    if len(batch) == 8:
                        chosen = batch
                        break
                if chosen:
                    break
            if chosen is None:
                raise ValueError(f"Insufficient eligible groups for {split}, class {cls}")
            for i, row in enumerate(chosen):
                role = "train" if split == "train" else ("query" if i < 2 else "target")
                selected.append(
                    {
                        **row,
                        "split": split,
                        "role": role,
                        "path": f"structures/{row['record_id']}.pdb",
                    }
                )
                for k in used:
                    used[k].add(row[k])
    return selected


def audit_splits(records):
    overlaps = {}
    for key in ("fold", "superfamily", "pdb", "aa_sha256", "shape_sha256", "structure_sha256"):
        sets = {
            s: {r[key] for r in records if r["split"] == s} for s in ("train", "validation", "test")
        }
        overlaps[key] = {
            f"{a}:{b}": sorted(sets[a] & sets[b]) for a, b in itertools.combinations(sets, 2)
        }
        assert not any(overlaps[key].values()), key
    return overlaps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--catalog-only", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "data-freeze.json").exists():
        raise FileExistsError("Dataset already frozen")
    data = read(out / "catalog.json") if (out / "catalog.json").exists() else catalog(out)
    counts = Counter(r["superfamily"] for r in data["accepted"])
    print(
        "eligible",
        len(data["accepted"]),
        "SF >=8",
        sum(n >= 8 for n in counts.values()),
        "excluded",
        data["exclusions"],
        flush=True,
    )
    if args.catalog_only:
        return
    records = select(data["accepted"], {"train": 32, "validation": 8, "test": 8})
    overlaps = audit_splits(records)
    needed = {r["archive_member"]: r for r in records}
    (out / "structures").mkdir(exist_ok=False)
    with tarfile.open(ARCHIVE, "r|gz") as archive:
        for member in archive:
            if member.name in needed:
                row = needed.pop(member.name)
                content = archive.extractfile(member).read()
                assert digest_bytes(content) == row["structure_sha256"]
                (out / row["path"]).write_bytes(content)
    assert not needed
    pairs = []
    groups = defaultdict(list)
    for row in records:
        if row["split"] != "test":
            groups[(row["split"], row["superfamily"])].append(row["record_id"])
    for (split, sf), ids in sorted(groups.items()):
        combinations = sorted(
            itertools.combinations(sorted(ids), 2), key=lambda p: order_key(":".join(p))
        )[:12]
        pairs.extend({"split": split, "superfamily": sf, "a": a, "b": b} for a, b in combinations)
    known = {}
    for name in ("pdbs_train.txt", "pdbs_val.txt"):
        known[name] = set((out / "references" / name).read_text().split())
    overlap_official = {
        split: {
            name: sorted(
                r["record_id"] for r in records if r["split"] == split and r["record_id"] in ids
            )
            for name, ids in known.items()
        }
        for split in ("train", "validation", "test")
    }
    save(out / "manifest.json", records)
    save(out / "pair_plan.json", pairs)
    save(
        out / "split_audit.json",
        {
            "cross_split_overlaps": overlaps,
            "official_published_split_overlap": overlap_official,
            "counts": dict(Counter(r["split"] for r in records)),
            "source_scope": (
                "SCOPe2.01 paper benchmark; held out from this training, not from Foldseek"
            ),
            "shape_fingerprint": "CA lower-triangle distances, rounded to 0.01 angstrom",
            "selection_code_sha256": sha(Path(__file__)),
        },
    )
    save(out / "environment.json", environment())
    plan = Path(__file__).resolve().parents[1] / "PLAN_11_15.md"
    (out / "plan-at-freeze.md").write_bytes(plan.read_bytes())
    save(
        out / "data-freeze.json",
        {
            "manifest_sha256": sha(out / "manifest.json"),
            "pair_plan_sha256": sha(out / "pair_plan.json"),
            "catalog_sha256": sha(out / "catalog.json"),
            "split_audit_sha256": sha(out / "split_audit.json"),
            "plan_sha256": sha(out / "plan-at-freeze.md"),
            "experiment_config_sha256": sha(out / "experiment_config.json"),
            "no_model_or_search_results_used": True,
        },
    )
    print("frozen", Counter(r["split"] for r in records), "pairs", len(pairs), flush=True)


if __name__ == "__main__":
    main()
