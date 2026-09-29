import itertools

import numpy as np
import pytest

from experiments.features import parse_alignment
from experiments.matrices import estimate_matrix
from experiments.metrics import query_metrics


def test_average_precision_and_recall_use_all_relevant_targets():
    scores = [
        {"target_id": "a", "raw_score": 5},
        {"target_id": "b", "raw_score": 4},
        {"target_id": "c", "raw_score": 3},
    ]
    result = query_metrics("q", scores, {"a", "c", "missing"})
    assert result["AP"] == pytest.approx((1 + 2 / 3) / 3)
    assert result["recall_at_1"] == 1 / 3
    assert result["recall_at_10"] == 2 / 3
    assert query_metrics("q", [], {"a"})["AP"] == 0


def test_tie_expected_ap_matches_enumerating_permutations():
    expected = []
    for perm in itertools.permutations("abc"):
        correct = seen = total = 0
        for target in perm:
            seen += 1
            if target in "ac":
                correct += 1
                total += correct / seen
        expected.append(total / 2)
    rows = [{"target_id": t, "raw_score": 10} for t in "abc"]
    assert query_metrics("q", rows, {"a", "c"})["tie_expected_AP"] == pytest.approx(
        np.mean(expected)
    )


def test_matrix_smoothing_symmetry_and_neutral_invalid_state():
    counts = np.zeros((20, 20), dtype=int)
    counts[0, 0] = counts[1, 1] = 8
    counts[0, 1] = counts[1, 0] = 2
    matrix, stats = estimate_matrix(counts)
    # Hand-computed joint/background: 20 observed + 400*0.5 prior counts.
    expected = round(2 * np.log2((8.5 / 220) / (20 / 220) ** 2))
    assert matrix[0, 0] == expected
    np.testing.assert_array_equal(matrix, matrix.T)
    assert not matrix[20].any() and not matrix[:, 20].any()
    assert stats["directed_counts"] == 20
    with pytest.raises(ValueError, match="symmetric"):
        estimate_matrix(np.eye(20) + np.triu(np.ones((20, 20)), 1))


def test_alignment_indices_follow_gaps_and_close_pair_marks():
    text = 'TM-score= 0.7\nTM-score= 0.8\n(" : " denotes residue pairs)\nAB-CD\n:  .:\nA-BCD\n'
    scores, pairs = parse_alignment(text, "ABCD", "ABCD")
    assert scores == [0.7, 0.8]
    np.testing.assert_array_equal(pairs, [[0, 0], [3, 3]])
    with pytest.raises(ValueError, match="correspondence"):
        parse_alignment(text, "ABCE", "ABCD")
