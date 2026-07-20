# CLUDA Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add faithful source-only PSE-CLUDA-TCN and source-initialized full PSE-CLUDA baselines with provenance and audit tests.

**Architecture:** A self-contained `methods/cluda` package adapts TimeMatch batches and implements official CLUDA behavior. `train.py` only constructs the compatible classifier/full trainer and preserves existing PSE-LTAE behavior.

**Tech Stack:** Python, PyTorch, pytest, existing TimeMatch loaders/evaluation.

---

### Task 1: Behavioral contract tests

**Files:** Create the eight requested `tests/test_cluda_*.py` files.

- [ ] Write tests for input adaptation, TCN reference arithmetic, full forward outputs, momentum, queue wrap, NNCL, augmentation, loss composition, target-label exclusion, and CPU optimizer step.
- [ ] Run `pytest tests/test_cluda_*.py -q` and confirm import failures because `methods.cluda` does not exist.

### Task 2: Pure model components

**Files:** Create `methods/cluda/tcn.py`, `mlp.py`, `nearest_neighbor.py`, `augmentations.py`, and `losses.py`.

- [ ] Implement causal two-convolution residual blocks with normal initialization, mask-aware four-view augmentation, cosine nearest neighbors, GRL, and weighted loss composition.
- [ ] Run focused component tests until green.

### Task 3: CLUDA models

**Files:** Create `methods/cluda/model.py`, `config.py`, `__init__.py`.

- [ ] Implement the raw-pixel PSE + frozen sinusoid adapter and supervised-compatible classifier.
- [ ] Implement query/key state, normalized query/prediction behavior, projector, discriminator, queues, wrapping enqueue, momentum update, diagonal contrastive targets, and exact batch NNCL.
- [ ] Run forward/state tests until green.

### Task 4: Full trainer and entry point

**Files:** Create `methods/cluda/trainer.py`; modify `train.py` and `dataset.py` only where needed.

- [ ] Build source/target loaders with unchanged TimeMatch transforms.
- [ ] Generate four independent views, compute five losses, log every required scalar, and save chronological checkpoints without target validation.
- [ ] Route `method == 'cluda'` directly to `train_cluda_full`; route source-only `--model cludatcn` through `train_supervised` with source validation.
- [ ] Run parser and training-step tests.

### Task 5: Provenance and launchers

**Files:** Create `methods/cluda/provenance.py`, `methods/cluda/PROVENANCE.md`, and `launchers/baselines/run_cluda_pse_screen.sh`.

- [ ] Record upstream URLs, inspected line logic, absent-license status, defaults, input adaptations, and exact four-group commands for three tasks/seed 1.
- [ ] Run shell syntax validation.

### Task 6: Final verification

- [ ] Run all CLUDA tests, then the repository test suite.
- [ ] Record max TCN error and deterministic first-step five-loss audit.
- [ ] Recheck target-label and target-validation call paths, `git diff --check`, and `git diff --stat`.
- [ ] Do not commit, merge, or run long experiments.
