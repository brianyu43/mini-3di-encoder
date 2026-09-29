"""Encode the audited frozen manifest, leaving prior artifacts untouched."""

from collections import Counter
from time import perf_counter

import numpy as np

from experiments.common import ALPHABET, ROOT, read, save, sha
from mini3di_encoder.atoms import read_backbone
from mini3di_encoder.encode import encode_chain, search_record
from mini3di_encoder.learned import load_learned_model, predict_states

from .common import ORIGINAL, OUT


def main():
    import json

    out = OUT
    manifest = read(out / "manifest-v2.json")
    freeze = read(out / "data-freeze.json")
    assert sha(out / "manifest-v2.json") == freeze["manifest_sha256"]
    directory = out / "encoded"
    directory.mkdir(exist_ok=False)
    learned_path = ORIGINAL / "models/seed-29/encoder.json"
    assert sha(learned_path) == freeze["learned_encoder_sha256"]
    layers, spec = load_learned_model(learned_path)
    files, total = [], 0.0
    counts = Counter()
    for row in manifest:
        if row["split"] == "train":
            continue
        sid = row["record_id"]
        source = out / row["path"]
        assert sha(source) == row["structure_sha256"]
        chain = read_backbone(source, row["chain"])
        assert chain.aa == row["aa"]
        began = perf_counter()
        official = encode_chain(chain, sid)
        x = np.array([r["features"] if r["valid"] else [0.0] * 10 for r in official["rows"]])
        states = predict_states(layers, spec, x)
        learned = search_record(official)
        learned["three_di"] = "".join(
            ALPHABET[s] if ok else "X"
            for s, ok in zip(states, official["valid_seed_mask"], strict=True)
        )
        total += perf_counter() - began
        np.savez_compressed(
            directory / f"{sid}.npz",
            x=x,
            mask=np.array(official["valid_seed_mask"]),
            official_states=np.array([r["state"] for r in official["rows"]]),
            learned_states=states,
            partners=np.array(
                [r["partner"] if r["partner"] is not None else -1 for r in official["rows"]]
            ),
        )
        for name, record in [("official", search_record(official)), ("learned", learned)]:
            assert all(
                (c != "X") == ok
                for c, ok in zip(record["three_di"], record["valid_seed_mask"], strict=True)
            )
            target = directory / f"{name}-{row['split']}-{row['role']}.jsonl"
            with target.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
        counts[row["split"]] += 1
        files.append(sid)
        if len(files) % 100 == 0:
            print("encoded", len(files), flush=True)
    save(
        out / "encoding.json",
        {
            "counts": dict(counts),
            "encode_seconds_sum": total,
            "learned_encoder_sha256": sha(learned_path),
            "script_sha256": sha(__file__),
            "geometry_code_sha256": sha(ROOT / "src/mini3di_encoder/geometry.py"),
            "files": {p.name: sha(p) for p in directory.iterdir()},
            "record_mask_checks": 2 * len(files),
        },
    )
    print("encoded all", len(files), flush=True)


if __name__ == "__main__":
    main()
