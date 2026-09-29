"""Build an outcome-independent hard-negative candidate manifest before similarity audit."""

import itertools
import shutil
import tarfile
from collections import Counter, defaultdict

from experiments.common import digest_bytes, read, save, sha
from experiments.prepare import ARCHIVE

from .common import ORIGINAL, OUT, order


def main():
    out = OUT
    destination = out / "candidate-manifest.json"
    if destination.exists():
        raise FileExistsError(destination)
    catalog = read(ORIGINAL / "catalog.json")["accepted"]
    old = read(ORIGINAL / "manifest.json")
    train = [dict(r) for r in old if r["split"] == "train"]
    old_folds = {r["fold"] for r in old}
    exposed = {key: {r[key] for r in old} for key in ["pdb", "aa_sha256", "shape_sha256"]}
    known = {}
    for key in ["train", "val"]:
        known[key] = set((ORIGINAL / f"references/pdbs_{key}.txt").read_text().split())
    known_pdb = {s[1:5].lower() for values in known.values() for s in values}
    known_sid = set.union(*known.values())
    # PDB/AA/geometry duplicates cannot be shared by selected records.
    unique_keys = ["pdb", "aa_sha256", "shape_sha256", "structure_sha256"]
    used = {key: {r[key] for r in train} for key in unique_keys}
    rows = train.copy()
    selection = []
    for split, query_goal in [("validation", 30), ("test", 100)]:
        groups = defaultdict(lambda: defaultdict(list))
        for row in catalog:
            fold = row["fold"]
            eligible = (
                fold in {"b.34", "c.26", "d.17"} if split == "validation" else fold not in old_folds
            )
            if split == "test" and any(row[key] in values for key, values in exposed.items()):
                eligible = False
            if eligible and not any(row[key] in used[key] for key in unique_keys):
                groups[fold][row["superfamily"]].append(row)
        batches = []
        for fold in sorted(groups, key=order):
            sfs = groups[fold]
            local = {key: values.copy() for key, values in used.items()}
            chosen = {}
            for sf in sorted(sfs, key=order):
                picked = []
                for row in sorted(sfs[sf], key=lambda r: order(r["record_id"])):
                    if any(row[key] in local[key] for key in unique_keys):
                        continue
                    picked.append(row)
                    for key in unique_keys:
                        local[key].add(row[key])
                if len(picked) >= 2:
                    chosen[sf] = picked
            eligible_sfs = [
                sf
                for sf, records in chosen.items()
                if len(records) >= 4 and len({r["family"] for r in records}) >= 2
            ]
            if len(chosen) < 2 or not eligible_sfs:
                continue
            pending = []
            for sf, records in chosen.items():
                targets, query_pool = [], []
                if sf in eligible_sfs:
                    # Reserve different-family targets; prefer list-absent records as queries.
                    normal = sorted(
                        records,
                        key=lambda r: (r["record_id"] not in known_sid, order(r["record_id"])),
                    )
                    targets.append(normal[0])
                    targets.append(next(r for r in normal if r["family"] != targets[0]["family"]))
                    rest = [r for r in records if r not in targets]
                    rest.sort(key=lambda r: (r["pdb"] in known_pdb, order(r["record_id"])))
                    query_pool = rest[:4]
                    targets += rest[4:12]
                else:
                    targets = records[:3]
                pending.append(
                    {"fold": fold, "sf": sf, "targets": targets, "query_pool": query_pool}
                )
            batches.extend(pending)
            for batch in pending:
                for row in batch["targets"] + batch["query_pool"]:
                    for key in unique_keys:
                        used[key].add(row[key])
        eligible = [b for b in batches if b["query_pool"]]
        selected_queries = []
        # Round robin spreads queries across SFs before adding a second query to one SF.
        for level in range(4):
            for batch in eligible:
                if level < len(batch["query_pool"]) and len(selected_queries) < query_goal:
                    selected_queries.append(batch["query_pool"][level]["record_id"])
        if len(selected_queries) != query_goal:
            raise ValueError(
                f"Only {len(selected_queries)} of {query_goal} {split} queries available"
            )
        selected_queries = set(selected_queries)
        for batch in batches:
            if not any(r["record_id"] in selected_queries for r in batch["query_pool"]):
                targets = batch["targets"][:3]
            else:
                targets = batch["targets"]
            for role, records in [("query", batch["query_pool"]), ("target", targets)]:
                for row in records:
                    if role == "query" and row["record_id"] not in selected_queries:
                        continue
                    rows.append(
                        {
                            **row,
                            "split": split,
                            "role": role,
                            "path": f"structures/{row['record_id']}.pdb",
                        }
                    )
        selection.append({"split": split, "query_goal": query_goal, "query_sfs": len(eligible)})
    for row in rows:
        row["published_sid_train"] = row["record_id"] in known["train"]
        row["published_sid_validation"] = row["record_id"] in known["val"]
        row["published_pdb_present"] = row["pdb"] in known_pdb
    assert len({r["record_id"] for r in rows}) == len(rows)
    denominators = []
    for query in (r for r in rows if r["role"] == "query"):
        targets = [r for r in rows if r["split"] == query["split"] and r["role"] == "target"]
        counts = {
            "positive": sum(t["superfamily"] == query["superfamily"] for t in targets),
            "cross_family_positive": sum(
                t["superfamily"] == query["superfamily"] and t["family"] != query["family"]
                for t in targets
            ),
            "same_fold_negative": sum(
                t["fold"] == query["fold"] and t["superfamily"] != query["superfamily"]
                for t in targets
            ),
            "other_fold_negative": sum(t["fold"] != query["fold"] for t in targets),
        }
        assert all(value > 0 for value in counts.values()), query["record_id"]
        assert len(targets) <= 1000
        denominators.append({"query_id": query["record_id"], "split": query["split"], **counts})
    for key in ["fold", "superfamily", *unique_keys]:
        sets = [
            {r[key] for r in rows if r["split"] == split}
            for split in ["train", "validation", "test"]
        ]
        assert all(not a & b for a, b in itertools.combinations(sets, 2)), key
    out.mkdir(exist_ok=True)
    save(destination, rows)
    save(out / "candidate-denominators.json", denominators)
    save(
        out / "selection-audit.json",
        {
            "catalog_sha256": sha(ORIGINAL / "catalog.json"),
            "previous_manifest_sha256": sha(ORIGINAL / "manifest.json"),
            "selection": selection,
            "counts": dict(Counter(r["split"] + ":" + r["role"] for r in rows)),
            "test_query_sfs": len(
                {r["superfamily"] for r in rows if r["split"] == "test" and r["role"] == "query"}
            ),
            "test_folds": len({r["fold"] for r in rows if r["split"] == "test"}),
            "test_published_pdb_absent_queries": [
                r["record_id"]
                for r in rows
                if r["split"] == "test" and r["role"] == "query" and not r["published_pdb_present"]
            ],
            "score_results_used": False,
            "script_sha256": sha(__file__),
        },
    )
    # Preserve previous model-selection data as exposure-audit references, not new test records.
    references = [r for r in old if r["split"] != "train"]
    save(out / "previous-exposure-manifest.json", references)
    needed = {r["archive_member"]: r for r in rows + references}
    (out / "structures").mkdir()
    with tarfile.open(ARCHIVE, "r|gz") as archive:
        for member in archive:
            if member.name in needed:
                row = needed.pop(member.name)
                blob = archive.extractfile(member).read()
                assert digest_bytes(blob) == row["structure_sha256"]
                (out / f"structures/{row['record_id']}.pdb").write_bytes(blob)
    assert not needed
    shutil.copyfile(ORIGINAL / "tools/usalign/TMalign", out / "TMalign")
    (out / "TMalign").chmod(0o755)
    print(read(out / "selection-audit.json"), flush=True)


if __name__ == "__main__":
    main()
