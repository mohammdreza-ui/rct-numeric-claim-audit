# Data statement

**What is published in this repository:** code, prompts, configs, PubMed IDs (PMIDs) and sampling seeds
(`data/manifest.csv`), human labels keyed by `claim_id` (`annotation/labels_*.csv`), and aggregate result tables
and figures (`results/`).

**What is NOT published:** abstract text, edited (counterfactual) abstracts, full model summaries, and the
local label sheets. PubMed abstracts can be subject to publisher copyright; anyone reproducing the study
re-downloads them from PubMed with `rctaudit fetch` using the shipped manifest of PMIDs (`rctaudit fetch --manifest data/manifest.csv`). Short atomic claims
produced by models are not published verbatim either unless checked for long verbatim overlap with abstracts.

**Sources:** abstracts from PubMed via NCBI E-utilities (respecting the 3 req/s limit, 10 req/s with a key,
and the NCBI usage policy). Optionally, PMIDs from the EBM-NLP corpus (Nye et al., ACL 2018) and/or the
EvidenceOutcomes corpus; check each dataset's own licence before redistributing anything derived from it
(the EBM-NLP Hugging Face card lists its licence as "unknown").

**Models:** all generation, claim extraction and verification run on local models (no abstract text is sent to
any third-party model API). No patient data, no private data of any kind is involved.

**Human participants / annotation:** the labels are by the repository author; no personal data.

**Known biases:** English-language RCT abstracts with numeric results only; PubMed query date window and
filters are in `configs/config.toml` and `src/rctaudit/sample.py`.
