# DECISIONS.md — microscope

Design decisions and their rationale. Each entry records what was chosen, the alternatives
considered, and why.

---

## Why layer 18 (not layer 5 or 25)

We train the SAE on `blocks.18.hook_resid_post` of Gemma-2-2B (26 layers total).

- **Layer 5 (early).** Early residual streams are dominated by token- and position-level
  features (detokenization, n-gram surface form). SAEs there recover lots of features but they
  are mostly lexical, not the higher-level semantic/relational structure we want for circuit
  analysis on a task like IOI.
- **Layer 25 (late).** The last couple of layers are heavily specialized toward the output
  (next-token) distribution; features become "logit-aligned" and less reusable as general
  concepts, and there is little downstream computation left to interpret.
- **Layer 18 (~70% depth).** Mid-to-late layers carry the most abstract, composed features
  while still feeding several blocks of downstream computation. This is the band where prior SAE
  work on similarly sized models finds the richest interpretable feature dictionaries, and it is
  deep enough that IOI-relevant name-mover / induction information is already present in the
  residual stream.

Layer 18 is a single fixed choice for this project; a fuller study would sweep layers.

---

## TopK vs ReLU SAE

We use a **TopK** SAE (`activation_fn_name = "topk"`, `k = 64`, `l1_coefficient = 0.0`).

- **ReLU + L1.** The classic sparse-autoencoder objective adds an L1 penalty on activations to
  induce sparsity. It works but couples reconstruction and sparsity through a single coefficient
  that needs tuning, suffers from activation **shrinkage** (L1 biases active features toward
  zero), and produces a variable, input-dependent number of active features.
- **TopK.** Keeping exactly the `k` largest pre-activations per token enforces an explicit,
  constant sparsity level (here 64 active features per token), removes the L1 shrinkage bias, and
  drops a hyperparameter (no L1 coefficient to tune — it is set to `0.0`, since L1 with TopK is a
  no-op). The reconstruction/sparsity trade-off becomes a single interpretable integer.

For a controlled comparison against a random baseline, the determinism of a fixed `k` is also
convenient: both SAEs operate at identical sparsity.

---

## Expansion factor 8 (not 16 or 32)

SAE width = `expansion_factor x d_model = 8 x 2304 = 18432` features.

- **Higher (16, 32).** Wider dictionaries can split features more finely and often push
  auto-interp scores up, but cost scales with width: more training compute, more dead features to
  manage, and more features to analyze. With a 2B-parameter model and a 2B-token budget, a 16-32x
  dictionary risks under-training many features within the token budget.
- **8x.** A standard, well-trodden width that fits the compute/token budget (`~14 h`, 2B tokens),
  keeps the dead-feature fraction controllable (target < 10%), and yields a feature count large
  enough for meaningful circuit and auto-interp analysis. It is the conservative default here;
  expansion is an obvious axis for a follow-up sweep.

---

## RelP over ACDC

For IOI circuit discovery we use **RelP** (relevance/attribution patching) rather than **ACDC**.

- **ACDC** performs iterative edge ablation, re-running the model for each candidate edge and
  greedily pruning to a threshold. It is principled but **expensive** — cost grows with the number
  of edges, and it requires many forward passes per discovery run.
- **RelP** computes relevance/attribution scores in a small number of passes (forward +
  attribution), making circuit discovery dramatically cheaper while remaining faithful enough for
  a well-characterized task like IOI. We still keep a standard **activation-patching baseline**
  (`compare_with_activation_patching`) so RelP's discovered circuit can be checked against a
  ground-truth-style patching attribution.

> Implementation note: the pinned `vendor/relp` submodule is **not** a standalone `relp` package
> (the `relp.circuit_discovery.find_circuit` / `relp.metrics.faithfulness_f1` API in CLAUDE.md
> does not exist). RelP ships as a **fork of TransformerLens** at `vendor/relp/TransformerLens`
> that adds Layer-wise Relevance Propagation. `relp_wrapper.py` therefore targets the real API:
> it loads the forked TransformerLens, sets `cfg.use_lrp = True` with
> `LRP_rules = ['LN-rule', 'Identity-rule', 'Half-rule']`, and computes relevance patching as in
> the upstream `demos/lrp_patching.ipynb`
> (`attribution = corrupted_relevance · (clean_act − corrupted_act)`). Faithfulness F1 is
> measured against the canonical Wang et al. IOI head circuit; the activation-patching baseline
> uses the fork's `transformer_lens.patching` helpers. Faithfulness F1 is only meaningful for a
> model with a known IOI circuit (GPT-2 small); for Gemma-2-2B the relevance attribution is still
> computed but there is no canonical ground-truth circuit to score against.

---

## Honest interpretation of the trained vs random gap

The randomized-weights control trains the **identical** SAE pipeline on a Gemma-2-2B with random
weights and compares mean auto-interp scores (`run_random_baseline`).

- A randomly initialized model still produces structured activations (the architecture, layernorm,
  and embeddings impose correlations), so its SAE will recover **some** features that score above
  zero on auto-interp. The random baseline is therefore **not** expected to be ~0 — it sets the
  floor for "interpretable-looking structure that is not about learned computation."
- The quantity that matters is the **gap** (`trained_mean_score - random_mean_score`) and its
  **bootstrap 95% CI** (`gap_ci_95`). The interpretability claim is only supported if the CI
  **excludes zero**. A large absolute trained score with a CI that overlaps zero would mean we are
  largely measuring auto-interp's tendency to find patterns in any structured data, not genuine
  learned features.
- Dead-feature fractions are reported for both (`trained_dead_frac`, `random_dead_frac`) because a
  trained SAE with many dead features can inflate the mean over surviving features; the gap must
  be read alongside these.
- The CI here comes from a **parametric** bootstrap over per-feature score statistics. It assumes
  approximately normal per-feature scores; a non-parametric bootstrap over the raw per-feature
  scores would be stronger and is the natural next step if the gap is marginal.
