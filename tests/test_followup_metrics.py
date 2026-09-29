"""The follow-up must not turn neutral family hits into false negatives."""

import numpy as np

from experiments.followup.metrics import cluster_interval, eligible_targets, evaluate


def fixture():
    q = {
        "record_id": "q",
        "family": "a.1.1.1",
        "superfamily": "a.1.1",
        "fold": "a.1",
        "published_sid_train": False,
        "published_sid_validation": False,
        "published_pdb_present": False,
    }
    targets = [
        dict(q, record_id="same"),
        dict(q, record_id="remote", family="a.1.1.2"),
        dict(q, record_id="hard", family="a.1.2.1", superfamily="a.1.2"),
        dict(q, record_id="easy", family="a.2.1.1", superfamily="a.2.1", fold="a.2"),
    ]
    return q, targets


def test_cross_family_excludes_neutral_targets_and_reranks():
    q, targets = fixture()
    scores = [
        {"query_id": "q", "target_id": t, "raw_score": s}
        for t, s in [("same", 40), ("hard", 30), ("remote", 20), ("easy", 10)]
    ]
    result = evaluate(scores, [q], targets)
    assert result["all"]["summary"]["MAP"] == (1 + 2 / 3) / 2
    assert result["cross_family"]["summary"]["MAP"] == 1 / 2
    assert result["cross_family"]["per_query"][0]["relevant_count"] == 1
    assert result["other_fold"]["summary"]["MAP"] == 1
    assert result["all"]["strata"]["published_pdb_absent"]["queries"] == 1


def test_hard_negative_view_keeps_the_positive_and_excludes_easy_negatives():
    q, targets = fixture()
    assert {r["record_id"] for r in eligible_targets(q, targets, "same_fold")} == {
        "same",
        "remote",
        "hard",
    }


def test_cluster_mean_preserves_unequal_query_counts():
    rows = [{"fold": "a", "AP": 1.0}] + [{"fold": "b", "AP": 0.0}] * 3
    result = cluster_interval(rows)
    assert result["mean"] == 0.25
    assert result["folds"] == 2 and result["queries"] == 4
    np.testing.assert_allclose(result["ci95"], [0, 1])


def test_missing_relevant_target_stays_in_denominator():
    q, targets = fixture()
    result = evaluate([{"query_id": "q", "target_id": "remote", "raw_score": 20}], [q], targets)
    assert result["all"]["summary"]["MAP"] == 0.5
    assert result["all"]["summary"]["recall_at_10"] == 0.5
    assert result["cross_family"]["summary"]["MAP"] == 1
