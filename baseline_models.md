# TRACE hack-detection baselines

Three baselines for the binary classification task: given a TRACE trajectory, predict
`is_hacked` (True = the assistant reward-hacked in ≥1 turn; False = benign).
Each baseline answers a different question:

1. **TF-IDF + LogReg** — what's the lexical floor?
2. **Frozen encoder + LogReg** — what do pretrained semantic representations add over n-grams?
3. **Fine-tuned encoder** — what does adapting representations to the task add over frozen ones?

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

## Token-length reference

Relevant numbers when choosing a baseline (cl100k_base BPE, per trajectory):

| representation | median | p95 | max |
|---|---|---|---|
| speech-only (just `content`) | 1,049 | 1,861 | 4,076 |
| speech + tool I/O (full)     | 10,665 | 18,916 | 24,478 |

78% of trajectories exceed 8K tokens with tool I/O included; all fit under 32K.

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
- Tiny model (~13M params dominated by the 50K → 256 input layer); fast to iterate.
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
Output is saved to `baseline_mlp_output.md` by default (use `--output-md PATH` to
change).

---

## Deferred — Long-context encoder approaches

A pretrained-encoder family of baselines (frozen embeddings + LogReg, full
fine-tuning) is implemented in `scripts/baseline_encoder.py` (dry-run validates the
pipeline) but not yet treated as part of the documented baseline set. They require
GPU/MPS for tractable training time and a ~600 MB checkpoint download. We'll revisit
once the TF-IDF baselines are nailed down and we know whether the lexical floor is
already close to whatever ceiling the task supports.

---

## Shared evaluation

Both scripts emit the same metrics on val and test via
`baseline_common.evaluate_predictions`:

- accuracy
- precision / recall / F1 for the `hacked` class
- ROC-AUC when the model produces scores (both do)
- confusion matrix
- predictions saved to `data/predictions/<baseline>_<split>.jsonl` with
  `(trajectory_id, y_true, y_pred, score, raw_output)` per row.

This makes it easy to compare models, ensemble them, or do error analysis by joining
predictions back to the processed trajectories.

---

## Suggested running order

1. **TF-IDF + LogReg** first (fast, free) → lexical floor; sanity-checks the pipeline.
2. **TF-IDF + MLP** second → does adding nonlinearity over the same features help, or
   does the small dataset just push us into overfitting? A small gain validates that
   feature interactions matter; a tie or regression argues the task is mostly linear
   in n-grams at this dataset size.
