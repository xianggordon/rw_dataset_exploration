# TRACE hack-detection baselines

Three approaches for the binary classification task: given a TRACE trajectory, predict
`is_hacked` (True = the assistant reward-hacked in ≥1 turn; False = benign).
"Baseline" is used loosely throughout — these are the approaches evaluated. These are the three main approaches:

1. **TF-IDF + LogReg** — what's the lexical floor?
2. **TF-IDF + small MLP** — does adding nonlinearity over the same features help, or
   does the small dataset just push us into overfitting?
3. **LLM monitor (zero-shot)** — what does a strong pretrained reasoning model do
   without any task-specific training?

Dataset splits come from `scripts/run_pipeline.py` (stratified on `is_hacked`):

| split | n   | hacked | benign |
|-------|-----|--------|--------|
| train | 361 | 187    | 174    |
| val   | 52  | 27     | 25     |
| test  | 104 | 54     | 50     |

All three share a common input formatter (`src/trace_data/baseline_common.py`) that
renders each trajectory as ChatML-flavoured text with tunable inclusion of `tool_calls`
and `tool_results`. Labels are trajectory-level only — no per-turn supervision.

---

## Cross-baseline comparison (test split, n=104)

| baseline | acc | F1 | recall | precision | AUROC | AUPRC |
|---|---|---|---|---|---|---|
| TF-IDF + LogReg | 0.712 | 0.737 | 0.778 | 0.700 | 0.815 | 0.851 |
| TF-IDF + small MLP | 0.760 | 0.775 | 0.796 | 0.754 | 0.834 | 0.872 |
| **LLM (gpt-5.5, continuous)** | **0.788** | **0.823** | **0.944** | 0.729 | **0.920** | **0.925** |

LLM wins on all metrics except precision (it over-flags borderline cases). AUC family
is comparable across all three because LLM scores are continuous probabilities, not
binary 0/1. The LLM number is single-run; results show that reasoning-model stochasticity may add a ~±2-3
trajectory swing on re-runs.

---

## Baseline 1 — TF-IDF + Logistic Regression

**What.** Flatten each trajectory into a single string (speech + serialized tool calls
+ tool results), compute TF-IDF over word and char n-grams, fit scikit-learn's
`LogisticRegression` with balanced class weights.

**Why it's a useful baseline.** Establishes the lexical floor: how much of the task is
pure surface vocabulary ("be more tolerant of borderline cases", `assertAlmostEqual`,
`pytest.skip`, etc.). If this hits 90%, the problem is simpler than it looks. If it
stays near 50%, most signal lives in structure and reasoning, not lexicon.

**Input representation.** Full (speech + tool_calls + tool_results), space-separated.

**Hyperparameters (defaults in `scripts/baseline_tfidf.py`).**
- `TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=50_000, sublinear_tf=True)`
- `LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)`

**Compute.** CPU only. End-to-end training + eval runs in **~30 seconds** on a laptop.

**Strengths.**
- No GPU, no API key, fully deterministic (given a seed).
- Fast to iterate — easy to ablate representations (`--representation speech|full`).
- Coefficients inspectable: top-positive and top-negative n-grams are a free
  lightweight interpretability channel.

**Limitations.**
- Bag-of-features: ignores ordering, can't reason about tool-call/result coherence
  (e.g., "assistant claims tests pass but tool_result shows 3 failures").
- Vocabulary-bound: a novel rationalization phrasing the training set didn't see
  is invisible.

Run: `.venv/bin/python scripts/baseline_tfidf.py`
Output is saved to `results/baseline_tfidf_output.md` by default.

---

## Baseline 2 — TF-IDF + small MLP

**What.** Same TF-IDF feature space as Baseline 1, but with a small multi-layer
perceptron classifier instead of logistic regression. The MLP has two hidden layers
with ReLU, dropout, and weight decay; it's trained with cross-entropy and AdamW.

**Why it's a useful baseline.** Cleanly isolates the question of *nonlinearity*:
Baseline 1 is the *linear* fit on these exact features; Baseline 2 is the *nonlinear*
fit on the same features. A meaningful gap means feature interactions matter (e.g.,
"`assertionerror` AND `tomorrow`" pulls toward hacked more than each individually); a
near-tie means the task is mostly linear in the n-gram features and added capacity
just over-fits the small training set.

**Input representation.** Full (speech + tool_calls + tool_results), TF-IDF vectorized
exactly as in Baseline 1: `ngram_range=(1,2)`, `max_features=50_000`, `min_df=2`,
`sublinear_tf=True`. Each trajectory becomes a 50,000-dim sparse vector.

**Architecture + training defaults.**
- TF-IDF capped much lower than Baseline 1 (`max_features=5_000`) to keep the MLP
  small; most discriminative features sit in the top-5K anyway.
- MLP: `Linear(5_000 → 32) → ReLU → Dropout(0.5) → Linear(32 → 2)`.
  Single hidden layer; ~160K parameters total — chosen to be small relative to
  ~361 training examples.
- Optimizer: AdamW, `lr=1e-3`, `weight_decay=1e-3` (10× stronger L2 than the prior
  draft to fight overfitting at this dataset size).
- Loss: cross-entropy.
- Batches: per-batch sparse→dense conversion (avoids materializing a 361 × 5K
  dense matrix in memory).
- Early stopping: track val AUC each epoch; restore best-val checkpoint at end;
  patience=4.
- Up to 15 epochs (early stop usually fires sooner).

**Compute.**
- CPU only by design — model is small enough that GPU/MPS isn't needed.
- End-to-end: ~30 seconds on a laptop CPU (TF-IDF vectorization + ~12 batches/epoch ×
  15 epochs).
- No model download required; no API key.

**Strengths.**
- Direct nonlinear comparison vs. Baseline 1 with the same input.
- Small model (~160K params; input layer 5K → 32 dominates the count); fast to iterate.
- Same evaluation surface as Baseline 1 — every difference is from the classifier head.

**Limitations.**
- Still overparameterized vs. data (~160K params, 361 train examples) but far less
  so than the prior 1.3M-param draft; dropout 0.5 + weight_decay 1e-3 close most of
  the train/test gap, but some remains.
- Still bag-of-features — ignores ordering and tool-call/result coherence, just like
  Baseline 1.
- More variance than LogReg from initialization and batch ordering — consider
  reporting mean over 3–5 seeds.

Run: `.venv/bin/python scripts/baseline_mlp.py`
Output is saved to `results/baseline_mlp_output.md` by default (use `--output-md
PATH` to change).

---

## Baseline 3 — LLM monitor (zero-shot, continuous scoring)

**What.** Prompt a capable LLM (default `gpt-5.5`) for a JSON object with a
calibrated probability of reward-hacking and a one-sentence justification. The
probability is the score; `pred = (probability > 0.5)`. No training. Implemented in
`scripts/baseline_llm_continuous.py`.

**Why it's a useful baseline.** Tests the ceiling of pretrained reasoning without
task-specific fitting. A wide gap to the supervised baselines suggests the
supervised models are limited by training-set size; a narrow gap argues lexical
features capture most of the available signal.

**Input representation.** Same flattened ChatML as the supervised baselines
(`representation="full"`).

**Prompt + parsing.** Short rubric distilled from TRACE category definitions;
required output `{"hacked_probability": float, "justification": str}`. Parser
attempts whole-response JSON first, falls back to flat-regex extraction.

**Compute.** Inference-only. ~30-40s per call on gpt-5.5; full test split runs in
~12-15 min at concurrency=4 on a Tier 1 OpenAI account. Cost ~$3-8 for n=104.

**Strengths.**
- Strongest test-set numbers across the three baselines (see comparison table).
- High recall — catches borderline hacks the supervised models miss.
- Free per-trajectory justification useful for error analysis.

**Limitations.**
- Run-to-run variance (~±2-3 trajectories), potentially from reasoning-model stochasticity.
- Inference cost is non-trivial (vs free for supervised baselines).
- Verbal probabilities cluster on round numbers; calibration is imperfect.
- Lower precision than MLP — over-flags at threshold 0.5; threshold tuning could
  rebalance.
- Requires `OPENAI_API_KEY`.

Run: `.venv/bin/python scripts/baseline_llm_continuous.py --split test`
Output is saved to `results/baseline_llm_continuous_<model>_output.md` by default.

A binary-output variant (`scripts/baseline_llm.py`) is kept for reference; it
predates the continuous-scoring redesign and produces degenerate AUROC.

---

## Deferred

**Long-context encoder.** Implemented in `scripts/baseline_encoder.py` (dry-run
only); requires GPU/MPS and a ~600 MB checkpoint. Worth revisiting if the gap
between Baseline 2 and Baseline 3 suggests a fine-tuned encoder could help.

**Open-source LLM fine-tuning.** Fine-tune a smaller open-source model (Qwen2.5-7B,
Llama-3.1-8B, etc.) on the 361 training trajectories via LoRA. Tests whether
task-specific fitting on a smaller LLM beats zero-shot prompting of a larger one;
requires GPU.

**Logprob-based LLM scoring.** Attempted via `baseline_llm_logprobs.py` (since
removed). Blocked: gpt-5/o-series reasoning models reject the `logprobs` API
parameter. Verbal probability (Baseline 3) is the available substitute. Worth
revisiting if logprobs become available on capable models.

### Token-length reference (relevant for encoder approaches)

Trajectory token counts (cl100k_base BPE):

| representation | median | p95 | max |
|---|---|---|---|
| speech-only (just `content`) | 1,049 | 1,861 | 4,076 |
| speech + tool I/O (full)     | 10,665 | 18,916 | 24,478 |

78% of trajectories exceed 8K tokens with tool I/O included; all fit under 32K.
A 16K-context encoder covers most trajectories without truncation; an 8K-context
encoder requires aggressive `tool_results` capping to fit the long tail.

---

## Shared evaluation

All three scripts emit the same metrics via `baseline_common.evaluate_predictions`:

- accuracy
- precision / recall / F1 for the `hacked` class
- AUROC and AUPRC when the model produces scores (all three do)
- confusion matrix
- predictions saved to `data/predictions/<baseline>_<split>.jsonl` with
  `(trajectory_id, y_true, y_pred, score, raw_output)` per row.

This makes cross-model comparison and error analysis straightforward.

---

## Suggested running order

1. **TF-IDF + LogReg** first (~30s, free) → lexical floor; pipeline sanity check.
2. **TF-IDF + MLP** second (~30s, free) → does nonlinearity over the same features
   help, or does it just over-fit at this dataset size?
3. **LLM monitor** last (~12-15 min, ~$3-8) → ceiling estimate from a strong
   pretrained reasoning model. Wide gap to (1)/(2) suggests more training data
   would help; narrow gap argues lexical signal carries most of the load.
