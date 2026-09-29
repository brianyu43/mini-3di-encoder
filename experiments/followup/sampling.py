"""Nested structure-pair sampling with stable superfamily coverage for R2."""

from collections import defaultdict

from .common import order


def nested_pair_subsets(pairs, counts=(38, 76, 152)):
    groups = defaultdict(list)
    for pair in pairs:
        if pair["split"] != "train" or not pair["accepted"]:
            raise ValueError("Only accepted training structure pairs may be sampled")
        groups[pair["superfamily"]].append(pair)
    if len({(p["a"], p["b"]) for p in pairs}) != len(pairs):
        raise ValueError("Duplicate structure pair")
    if not counts or min(counts) < len(groups) or max(counts) > len(pairs):
        raise ValueError("Subset counts must cover every group and fit the training pool")
    for sf in groups:
        groups[sf].sort(key=lambda p: order("R2:" + p["a"] + ":" + p["b"]))
    sequence = []
    for level in range(max(map(len, groups.values()))):
        for sf in sorted(groups, key=lambda sf: order("R2:" + sf)):
            if level < len(groups[sf]):
                sequence.append(groups[sf][level])
    return {count: sequence[:count] for count in counts}
