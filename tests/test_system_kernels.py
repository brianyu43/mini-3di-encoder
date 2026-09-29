"""Randomized differential tests against the unchanged search reference."""

import sys

import numpy as np

from experiments.common import ROOT, SEARCH_ROOT
from experiments.followup.system_kernels import (
    decoded_alignment,
    packed_postings,
    support_kernel,
    traceback_kernel,
    ungapped_kernel,
)

sys.path.insert(0, str(SEARCH_ROOT / "src"))
from mini3di_search.align_reference import align  # noqa: E402
from mini3di_search.index import IndexConfig, build_index  # noqa: E402
from mini3di_search.prefilter import (  # noqa: E402
    collect_hits,
    filter_ungapped,
    supported_diagonals,
)
from mini3di_search.records import Alphabet, ProteinRecord, Sequence  # noqa: E402
from mini3di_search.scoring import Scoring, load_matrix  # noqa: E402
from mini3di_search.search_numba import prepare_search  # noqa: E402
from mini3di_search.traceback import rescore_alignment  # noqa: E402


def scoring(go=14, ge=2):
    return Scoring(
        load_matrix(
            ROOT / "models/paired-vqvae-small/substitution.mat",
            kind=Alphabet.THREE_DI,
            source="fixed test matrix",
            synthetic=False,
        ),
        go,
        ge,
    )


def test_compiled_traceback_matches_ties_gaps_and_empty_paths():
    rng = np.random.default_rng(9182)
    for go, ge in [(14, 2), (1, 0), (0, 0)]:
        sc = scoring(go, ge)
        matrix = np.array(sc.matrix.values, dtype=np.int64)
        for _ in range(100):
            q = "".join(rng.choice(list("ACDEFX"), size=rng.integers(0, 35)))
            t = "".join(rng.choice(list("ACDEFX"), size=rng.integers(0, 35)))
            qs, ts = Sequence(q, Alphabet.THREE_DI), Sequence(t, Alphabet.THREE_DI)
            qa, ta = (
                np.array(sc.matrix.encode(qs), dtype=np.int64),
                np.array(sc.matrix.encode(ts), dtype=np.int64),
            )
            got = decoded_alignment(q, t, traceback_kernel(qa, ta, matrix, go, ge))
            assert got == align(qs, ts, sc)
            assert rescore_alignment(qs, ts, got, sc) == got.raw_score


def test_array_filters_match_reference_with_repeats_masks_and_diagonals():
    rng = np.random.default_rng(451)
    sc = scoring()
    for trial in range(35):
        rows = []
        for i in range(9):
            letters = rng.choice(list("ACD"), size=rng.integers(15, 90))
            mask = rng.random(len(letters)) > 0.12
            letters[~mask] = "X"
            text = "".join(letters)
            rows.append(
                ProteinRecord(f"r{i}", "A" * len(text), text, tuple(map(bool, mask)), False)
            )
        q, targets = rows[0], rows[1:]
        index = build_index(targets, IndexConfig(3), allow_real=True)
        p = prepare_search([q], targets, sc, allow_real=True)
        po, postings = packed_postings(index)
        support = supported_diagonals(collect_hits(q, index), k=3, window=64, double=True)
        ids, diagonals = support_kernel(
            p.query_arrays[0],
            np.array(q.valid_seed_mask),
            po,
            postings,
            max(len(t.three_di) for t in targets),
        )
        got = {}
        for tid, diagonal in zip(ids, diagonals, strict=True):
            got.setdefault(int(tid), []).append(int(diagonal))
        assert {k: tuple(v) for k, v in got.items()} == support, trial
        for threshold in [0, 10, 20, 100]:
            expected, scores = filter_ungapped(q, index, support, sc, threshold)
            passed, values = ungapped_kernel(
                p.query_arrays[0], p.flat_targets, p.offsets, ids, diagonals, p.matrix, threshold
            )
            assert tuple(passed) == expected
            assert list(values) == [scores[t] for t in expected]
