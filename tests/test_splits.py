from collections import defaultdict

from rctaudit.splits import dedupe_abstracts, group_folds, jaccard

A = "Randomised trial of drug X versus placebo in 120 adults with hypertension; systolic blood pressure fell by 12 mmHg."


def test_dedupe_near_duplicate_and_same_nct():
    recs = [{"pmid": "1", "abstract": A},
            {"pmid": "2", "abstract": A + " (extended report)"},
            {"pmid": "3", "abstract": "Completely different trial of drug Y in asthma NCT01234567."},
            {"pmid": "4", "abstract": "Follow up of another cohort. Registered as NCT01234567."}]
    kept, dropped = dedupe_abstracts(recs, threshold=0.7)
    assert [r["pmid"] for r in kept] == ["1", "3"]
    assert {d[1] for d in dropped} == {"2", "4"}
    assert jaccard(set(), {1}) == 0.0


def test_group_folds_keep_groups_together_and_are_deterministic():
    groups = [str(i // 4) for i in range(80)]  # 20 groups of 4 claims
    f1, f2 = group_folds(groups, 5, seed=1), group_folds(groups, 5, seed=1)
    assert f1 == f2 and f1 != group_folds(groups, 5, seed=2)
    by = defaultdict(set)
    for g, f in zip(groups, f1):
        by[g].add(f)
    assert all(len(v) == 1 for v in by.values())
    sizes = [f1.count(k) for k in range(5)]
    assert max(sizes) - min(sizes) <= 4


def test_group_folds_stratified_balances_strata():
    groups = [str(i) for i in range(100)]
    strata = ["a" if i < 50 else "b" for i in range(100)]
    folds = group_folds(groups, 5, 0, strata)
    for f in range(5):
        n_a = sum(1 for i in range(100) if folds[i] == f and strata[i] == "a")
        assert 5 <= n_a <= 15
