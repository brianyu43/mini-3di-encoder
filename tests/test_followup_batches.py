"""Batching preserves exhaustive pair coverage without lifting engine safety limits."""

from types import SimpleNamespace

import pytest

from experiments.followup.search import query_batches


def test_batches_preserve_order_and_exact_budget_boundary():
    queries = [SimpleNamespace(three_di="AAA", record_id=str(i)) for i in range(5)]
    batches = list(query_batches(queries, 10, limit=60))
    assert [len(b) for b in batches] == [2, 2, 1]
    assert [q for batch in batches for q in batch] == queries
    assert all(sum(len(q.three_di) for q in batch) * 10 <= 60 for batch in batches)


def test_single_oversize_query_is_rejected():
    with pytest.raises(ValueError, match="One query"):
        list(query_batches([SimpleNamespace(three_di="AAAA")], 10, limit=30))
