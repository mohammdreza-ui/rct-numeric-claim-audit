"""RQ1 + RQ2 analysis from human labels. Run:  python analysis/analyze.py --generator local

Inputs  (all by claim_id):
  annotation/labels_<gen>.csv          claim_id,claim_sha,label[,error_origin][,notes]  (write it with
                                       scripts/export_labels.py; every label is checked against the CURRENT claim
                                       text via claim_sha and the run is refused on a mismatch; error_origin in
                                       {summary, extractor}: claims whose error was introduced by the CLAIM
                                       EXTRACTOR are excluded from the generator audit and counted separately)
  results/numeric_<gen>.csv            rule-based verdict
  results/signals_<gen>.csv            logprob / self-consistency
  results/verbalized_<gen>.csv         optional
  results/pyes_<gen>.csv               optional (P(Yes) verifier prompt)
  results/minicheck_<gen>.csv          optional
  work/claims_<gen>.jsonl              for pmid / has_number / type
Outputs: results/rq1_<gen>.csv, results/rq2_<gen>.csv, results/rq2_<gen>.png
Everything is grouped by abstract (cluster bootstrap on PMID); calibration maps are fitted
cross-fold on group folds, never on the claims they are evaluated on.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rctaudit import metrics as M  # noqa: E402
from rctaudit.labels import EXCLUDED, LABELS, is_supported, verify_label_rows  # noqa: E402
from rctaudit.claims import claim_stratum  # noqa: E402
from rctaudit.splits import group_folds  # noqa: E402


def read_csv(path):
    p = Path(path)
    return list(csv.DictReader(open(p))) if p.exists() else []


def fnum(x):
    try:
        return float(x) if x not in ("", None, "None") else None
    except ValueError:
        return None


def check_labels(lab_rows, claims, allow_unverified: bool = False) -> None:
    """Stop if a label does not belong to the claim now stored under its claim_id. claim_id is the position
    of a claim in the extractor's output, so after a new extractor prompt/model the same id is a DIFFERENT claim
    and a plain join silently attaches labels to the wrong claims."""
    prob = verify_label_rows(lab_rows, claims)
    fatal = prob["missing"] + prob["changed"] + ([] if allow_unverified else prob["unverifiable"])
    if not fatal:
        if prob["unverifiable"]:
            print(f"WARNING: {len(prob['unverifiable'])} labels carry neither claim_sha nor claim text and were NOT verified")
        return
    n = len(lab_rows)
    lines = [f"{len(prob['changed'])} of {n} labels belong to a DIFFERENT claim than the one now stored under their claim_id "
             f"(e.g. {prob['changed'][:3]})" if prob["changed"] else None,
             f"{len(prob['missing'])} labels point to claim_ids that no longer exist (e.g. {prob['missing'][:3]})"
             if prob["missing"] else None,
             f"{len(prob['unverifiable'])} labels have no claim_sha / claim text to verify against (e.g. "
             f"{prob['unverifiable'][:3]}); re-export them with scripts/export_labels.py, or pass --allow-unverified"
             if prob["unverifiable"] and not allow_unverified else None]
    sys.exit("REFUSING to analyse: the labels do not match the current claims.\n  " +
             "\n  ".join(l for l in lines if l) +
             "\nThe claims were probably re-extracted (new prompt or model) after the sheet was labelled. Re-label the "
             "current work/label_sheet_*.csv, or restore the claims file the labels were made on.")


def load(gen: str, allow_unverified: bool = False):
    lab_rows = [r for r in read_csv(f"annotation/labels_{gen}.csv") if r["label"].strip()]
    labels = {r["claim_id"]: r["label"].strip() for r in lab_rows}
    origin = {r["claim_id"]: (r.get("error_origin") or "").strip().lower() for r in lab_rows}
    claims = {}
    for line in Path(f"work/claims_{gen}.jsonl").read_text().splitlines():
        for c in json.loads(line)["claims"]:
            claims[c["claim_id"]] = c
    check_labels(lab_rows, claims, allow_unverified)
    tables = {n: {r["claim_id"]: r for r in read_csv(f"results/{n}_{gen}.csv")}
              for n in ("numeric", "signals", "verbalized", "minicheck", "pyes")}
    rows = []
    for cid, lab in labels.items():
        if lab in EXCLUDED or cid not in claims:
            continue
        c, sig = claims[cid], tables["signals"].get(cid, {})
        rows.append({
            "claim_id": cid, "pmid": c["pmid"], "has_number": c["has_number"], "label": lab,
            "stratum": claim_stratum(c["text"]), "origin": origin.get(cid, ""),
            "y": is_supported(lab),
            "numeric_verdict": tables["numeric"].get(cid, {}).get("numeric_verdict"),
            "num_lp_min": fnum(sig.get("num_lp_min")), "num_lp_mean": fnum(sig.get("num_lp_mean")),
            "span_lp_mean": fnum(sig.get("span_lp_mean")),
            "bind_lp_min": fnum(sig.get("bind_lp_min")), "bind_lp_mean": fnum(sig.get("bind_lp_mean")),
            "sc_agree": fnum(sig.get("sc_agree")),
            "self_consistency": fnum(sig.get("self_consistency")),
            "p_yes": fnum(tables["pyes"].get(cid, {}).get("p_yes")),
            "verbalized": fnum(tables["verbalized"].get(cid, {}).get("verbalized")),
            "minicheck": fnum(tables["minicheck"].get(cid, {}).get("minicheck")),
        })
    return rows


def pool_strata_shares(gen: str) -> dict[str, float] | None:
    """Share of each claim stratum in the pool the label sheet was drawn from (claims whose span is in the
    summary and that are not fragments, see pipeline.sample_for_labelling). The sheet over-samples the hard
    numeric stratum, so rates over all claims must be re-weighted to these shares."""
    p = Path(f"work/claims_{gen}.jsonl")
    if not p.exists():
        return None
    pool = [c for line in p.read_text().splitlines() for c in json.loads(line)["claims"]
            if c.get("span_in_summary", True) and not c.get("extra", {}).get("degenerate")]
    if not pool:
        return None
    cnt = Counter(claim_stratum(c["text"]) for c in pool)
    return {k: v / len(pool) for k, v in cnt.items()}


def weighted_unsupported(y: np.ndarray, strata: np.ndarray, idx: np.ndarray, shares: dict[str, float]) -> float:
    """Post-stratified unsupported rate: sum_k share_k * rate_k, over strata present in the (re)sample;
    shares are renormalised over the present strata so a bootstrap replicate missing one stratum stays defined."""
    yi, si = y[idx], strata[idx]
    num = den = 0.0
    for k, w in shares.items():
        m = si == k
        if m.any():
            num += w * (1 - yi[m].mean())
            den += w
    return float(num / den) if den else float("nan")


def rq1(rows, gen, n_boot, pool_shares=None):
    out = []
    pm = np.array([r["pmid"] for r in rows])
    y = np.array([r["y"] for r in rows])
    cnt = Counter(r["label"] for r in rows)
    k, n = int((1 - y).sum()), len(y)
    strata = np.array([r.get("stratum") or ("numeric_hard" if r["has_number"] else "non_numeric") for r in rows])
    num = np.array([bool(r["has_number"]) for r in rows])
    sheet_shares = {s: float((strata == s).mean()) for s in set(strata)}
    shares = pool_shares or sheet_shares
    shares = {s: w for s, w in shares.items() if s in sheet_shares}  # strata with no labelled claim cannot be weighted
    pt, lo, hi = M.cluster_bootstrap(lambda i: weighted_unsupported(y, strata, i, shares), pm, n_boot)
    out.append({"quantity": "unsupported_rate_all", "estimate": pt, "ci_lo": lo, "ci_hi": hi, "n": n,
                "note": "re-weighted to the claim-pool strata " +
                        ", ".join(f"{s}={w:.2f}" for s, w in sorted(shares.items())) +
                        f"; cluster bootstrap by abstract. Raw sheet rate {k / n:.3f}, Wilson {M.wilson_ci(k, n)} "
                        "(ignores weighting and clustering)"})
    for s in sorted(sheet_shares):
        m = strata == s
        pt_, lo_, hi_ = M.cluster_bootstrap(
            lambda i, m=m: float(1 - y[m][i].mean()) if len(i) else float("nan"), pm[m], n_boot)
        out.append({"quantity": f"unsupported_rate_{s}", "estimate": pt_, "ci_lo": lo_, "ci_hi": hi_,
                    "n": int(m.sum()), "note": f"{int((1 - y[m]).sum())} unsupported"})
    if num.any():
        pt_, lo_, hi_ = M.cluster_bootstrap(
            lambda i: float(1 - y[num][i].mean()) if len(i) else float("nan"), pm[num], n_boot)
        out.append({"quantity": "unsupported_rate_numeric_claims", "estimate": pt_, "ci_lo": lo_, "ci_hi": hi_,
                    "n": int(num.sum()), "note": "all numeric strata pooled, NOT re-weighted between strata"})
    for lab in LABELS:
        if lab in EXCLUDED:
            continue
        lo_, hi_ = M.wilson_ci(cnt.get(lab, 0), n)
        out.append({"quantity": f"share_{lab}", "estimate": cnt.get(lab, 0) / n, "ci_lo": lo_, "ci_hi": hi_,
                    "n": n, "note": "Wilson (descriptive); share of the label sheet, NOT re-weighted"})
    # rule-based verifier vs human on numeric claims it can rule on
    ev = [(r["numeric_verdict"], r["y"]) for r in rows if r["numeric_verdict"] in ("supported", "unsupported")]
    if ev:
        a = [v for v, _ in ev]
        b = ["supported" if h else "unsupported" for _, h in ev]
        out.append({"quantity": "rule_vs_human_kappa", "estimate": M.cohen_kappa(a, b), "ci_lo": "", "ci_hi": "",
                    "n": len(ev), "note": "claims with a rule verdict of supported/unsupported"})
        fn = sum(1 for v, h in ev if v == "supported" and h == 0)
        out.append({"quantity": "rule_supported_but_human_unsupported", "estimate": fn / max(1, sum(v == 'supported' for v, _ in ev)),
                    "ci_lo": "", "ci_hi": "", "n": sum(v == 'supported' for v, _ in ev),
                    "note": "= misattribution/direction errors the number-presence check cannot see"})
    return out


SIGNALS = [("num_lp_min", "min logprob (number tokens)"), ("num_lp_mean", "mean logprob (number tokens)"),
           ("bind_lp_min", "min logprob (words around the number)"),
           ("bind_lp_mean", "mean logprob (words around the number)"),
           ("span_lp_mean", "mean logprob (sentence)"), ("verbalized", "verbalized 0-100"),
           ("p_yes", "P(Yes) verifier prompt"),
           ("sc_agree", "self-consistency (contradiction-aware)"),
           ("self_consistency", "numeric self-consistency (presence; ablation)"),
           ("minicheck", "MiniCheck score")]


MIN_MINORITY = 10  # fewer unsupported (or supported) claims than this: AUROC is descriptive only


def _eval_signal(sub, key, name, subset, n_boot, n_bins, seed):
    s = np.array([r[key] for r in sub], float)
    y = np.array([r["y"] for r in sub])
    pm = np.array([r["pmid"] for r in sub])
    n_bad = int((1 - y).sum())
    pt, lo, hi = M.cluster_bootstrap(lambda i: M.auroc(s[i], y[i]), pm, n_boot, seed)
    # cross-fitted calibration on group folds
    folds = np.array(group_folds(list(pm), 5, seed))
    p_platt, p_iso = np.zeros(len(s)), np.zeros(len(s))
    for f in range(5):
        tr, te = folds != f, folds == f
        if te.sum() == 0 or len(set(y[tr])) < 2:
            p_platt[te] = y[tr].mean() if tr.any() else 0.5
            p_iso[te] = p_platt[te]
            continue
        a_, b_ = M.fit_platt(s[tr], y[tr])
        p_platt[te] = M.apply_platt(s[te], a_, b_)
        p_iso[te] = M.fit_isotonic(s[tr], y[tr])(s[te])
    note = "ECE equal-mass, few bins; small n => wide uncertainty"
    if min(n_bad, len(y) - n_bad) < MIN_MINORITY:
        note = f"ONLY {min(n_bad, len(y) - n_bad)} claims in the minority class: AUROC/ECE are descriptive, not evidence. " + note
    return {
        "signal": name, "subset": subset, "n": len(sub), "n_unsupported": n_bad, "n_abstracts": len(set(pm)),
        "pos_rate_supported": float(y.mean()),
        "auroc": pt, "auroc_lo": lo, "auroc_hi": hi,
        "aurc": M.aurc(s, y), "risk@80%cov": M.risk_at_coverage(s, y, 0.8),
        "ece_platt": M.ece(p_platt, y, n_bins), "ece_isotonic": M.ece(p_iso, y, n_bins),
        "brier_platt": M.brier(p_platt, y), "note": note,
    }


def rq2(rows, n_boot, n_bins, seed):
    """Two views per signal. 'own': every labelled claim the signal exists for (logprob/verbalized/
    self-consistency exist only for NUMERIC claims, span logprob for most claims), so signals are NOT
    comparable across rows of this view. 'common': only claims for which every available signal exists;
    compare signals ONLY here."""
    out, curves = [], {}
    avail = [(k, nm) for k, nm in SIGNALS if sum(r[k] is not None for r in rows) >= 30]
    common = [r for r in rows if all(r[k] is not None for k, _ in avail)] if avail else []
    for key, name in SIGNALS:
        for subset, pool in (("own", rows), ("common", common)):
            sub = [r for r in pool if r[key] is not None]
            if (subset == "common" and (key, name) not in avail) or len(sub) < 30 or len({r["y"] for r in sub}) < 2:
                if subset == "own":
                    out.append({"signal": name, "subset": subset, "n": len(sub),
                                "note": "too few labelled claims / one class only / signal missing"})
                continue
            out.append(_eval_signal(sub, key, name, subset, n_boot, n_bins, seed))
            if subset == "common":
                s = np.array([r[key] for r in sub], float)
                curves[name] = M.risk_coverage(s, np.array([r["y"] for r in sub]))
    if not curves:  # nothing on a common subset: fall back to each signal's own claims
        for key, name in SIGNALS:
            sub = [r for r in rows if r[key] is not None]
            if len(sub) >= 30 and len({r["y"] for r in sub}) == 2:
                curves[name] = M.risk_coverage(np.array([r[key] for r in sub], float), np.array([r["y"] for r in sub]))
    return out, curves


def write(rows, path):
    if not rows:
        return
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def plot(curves, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not installed, figure skipped (pip install -e '.[analysis]')")
        return
    if not curves:
        print("WARNING: no signal had enough labelled claims, figure skipped")
        return
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for name, (cov, risk) in curves.items():
        ax.plot(cov, risk, label=f"{name} (n={len(cov)})", lw=1.5)
    ax.set_xlabel("coverage (fraction of claims kept)")
    ax.set_ylabel("error rate among kept claims")
    ax.set_title("Risk-coverage (pessimistic ties); n differs per signal, see rq2 csv", fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bins", type=int, default=5)
    ap.add_argument("--allow-unverified", action="store_true",
                    help="accept labels without claim_sha/claim text (old files); mismatches are still refused")
    a = ap.parse_args()
    all_rows = load(a.generator, a.allow_unverified)
    if not all_rows:
        sys.exit("No labelled claims found: fill annotation/labels_<gen>.csv first.")
    rows = [r for r in all_rows if r["origin"] != "extractor"]
    n_ext = len(all_rows) - len(rows)
    print(f"{len(all_rows)} labelled claims from {len({r['pmid'] for r in all_rows})} abstracts; "
          f"{n_ext} excluded as extractor-origin errors")
    t1 = rq1(rows, a.generator, a.boot, pool_strata_shares(a.generator))
    t1.append({"quantity": "claims_excluded_extractor_origin", "estimate": n_ext / len(all_rows), "ci_lo": "",
               "ci_hi": "", "n": len(all_rows),
               "note": "labelled claims whose error was introduced by the claim extractor, not the summary "
                       "(error_origin = extractor); the audit of the GENERATOR excludes them"})
    write(t1, f"results/rq1_{a.generator}.csv")
    t2, curves = rq2(rows, a.boot, a.bins, a.seed)
    write(t2, f"results/rq2_{a.generator}.csv")
    plot(curves, f"results/rq2_{a.generator}.png")
    print("wrote results/rq1_*.csv, rq2_*.csv, rq2_*.png")


if __name__ == "__main__":
    main()
