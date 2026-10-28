# Annotation guidelines (v1)

Goal: decide, for each atomic claim from an LLM summary, whether the **source abstract supports it**.
You see: the abstract, the summary, `span` (the summary sentence the claim was extracted from) and one claim. You do **not** see any automatic verdict or
confidence (the label sheet hides them on purpose, to avoid anchoring).

## Labels (exactly one per claim)

| Label | Use when |
|---|---|
| `Supported` | Every number, direction and qualifier in the claim is stated in the abstract (rounding of a stated number is fine, e.g. 14.5% -> 15%). |
| `Derived-correct` | The claim contains a number/direction that is NOT stated but follows by simple, correct arithmetic from stated numbers *for the right arms/outcome* (e.g. difference of two stated means; relative reduction from two stated event rates). |
| `Derived-wrong` | Same kind of derived number, but the arithmetic or the arms/outcome are wrong. |
| `Unsupported-number` | A number in the claim appears nowhere in the abstract and is not a correct derivation. |
| `Misattributed` | Every number exists in the abstract, but it is attached to the wrong arm, outcome, time point or population. |
| `Direction-wrong` | The claim states the opposite direction/sign of effect (benefit vs harm, higher vs lower, significant vs not significant). |
| `Unsupported-claim` | A **non-numeric** statement (a finding, a safety or tolerability result, an outcome, a population or intervention detail) that the abstract does not make at all, e.g. "No serious complications were reported" when the abstract only says complications were compared. If the abstract says something similar but weaker or narrower, use `Overgeneralized` instead. Numbers that are not in the abstract are `Unsupported-number`. |
| `Overgeneralized` | The claim is stronger or broader than the abstract: "proves", "cures", extends to a population/dose/duration not studied, turns "no significant difference" into "equally effective", causal wording for a non-causal finding. |
| `Skip` | Not a claim (fragment such as "The p=0.038.", extractor garbage). Skips are counted and reported, not hidden. Obvious fragments are already filtered out of the sheet. |

**`error_origin` (fill it for every label other than `Supported`, `Derived-correct` and `Skip`).** The claim is cut out of
the summary by a claim extractor (itself an LLM), which can break a correct sentence, e.g. split "23% and 39% in the
test and control groups, respectively" into two claims that each say "in the test and control groups". Read `span`:
- `summary`: the summary sentence itself is not supported by the abstract (the generator made the error);
- `extractor`: the summary sentence is fine, but the claim lost or changed something (the extractor made the error).
  These claims are excluded from the audit of the generator and counted separately.
Judge the **claim** against the abstract for `label` exactly as before; `error_origin` only says where the error came from.

If a claim has two problems, choose the **first applicable** row in this order:
`Direction-wrong` > `Misattributed` > `Unsupported-number` > `Unsupported-claim` > `Derived-wrong` > `Overgeneralized`.

## Rules of thumb
- Judge against the **abstract only**, not against your medical knowledge or the full paper.
- Tolerance: a stated number may be rounded; do **not** accept added precision (45 -> 45.3).
- "Not significant" means P >= .05 or a CI including the null as the abstract defines it; do not re-test.
  If the abstract itself calls a result "significant" with P >= .05 (it happens), the abstract's own wording wins for
  `Direction-wrong`: judge the claim against what the abstract says, and write the inconsistency in `notes`.
- Hedging words ("may", "suggests") that the abstract also uses are fine; hedges removed are `Overgeneralized`.
- Unsure? Apply the **clinician test**: would a clinician reading only the abstract sign off on this sentence? If not, it is not `Supported`. Write the doubt in `notes`.
- Do not look at earlier labels while labelling; do not edit labels after the test-retest.

## Protocol
1. One pass over ~200 claims (sampled by `rctaudit labelsheet`, <= 3 claims per abstract; half are "hard" numeric claims
   with percentages, decimals or comparisons, a minority are bare integers, the rest have no number; the analysis re-weights).
2. **Test-retest**: after >= 48 h, relabel a random 30 of them blind to your first labels. Report Cohen's kappa
   (`rctaudit.metrics.cohen_kappa`) on the 8-way labels and on supported/unsupported. A single annotator is a
   limitation and is stated as such in the README; a second annotator (even for 50 claims) is the best upgrade.
3. Do not edit `annotation/labels_<generator>.csv` by hand and never save the sheet itself there (it contains abstracts).
   Run `python scripts/export_labels.py`: it writes the PUBLIC file with columns `claim_id,claim_sha,label,error_origin,notes`
   (`claim_sha` ties each label to the exact claim text; the analysis refuses labels whose claim changed)
   (claim text and abstracts stay in the local, git-ignored label sheet).
