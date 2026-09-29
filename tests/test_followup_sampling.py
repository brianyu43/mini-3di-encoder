"""Sampling must preserve groups, nesting and strict train-only boundaries."""

import pytest

from experiments.followup.sampling import nested_pair_subsets


def test_nested_fraction_sampling_covers_groups_and_is_input_order_invariant():
    pairs = [
        {"a": f"a{i}", "b": f"b{i}", "superfamily": str(i % 3), "split": "train", "accepted": True}
        for i in range(12)
    ]
    a = nested_pair_subsets(pairs, (3, 6, 12))
    assert a == nested_pair_subsets(list(reversed(pairs)), (3, 6, 12))
    assert {p["superfamily"] for p in a[3]} == {"0", "1", "2"}
    assert a[3] == a[6][:3] and a[6] == a[12][:6]
    assert len(a[12]) == 12


def test_validation_pairs_cannot_enter_learning_curve():
    with pytest.raises(ValueError, match="training"):
        nested_pair_subsets([{"split": "validation", "accepted": True}], (1,))
