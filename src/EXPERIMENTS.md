# Action-dependence baselines and sensitivity experiments

These experiments leave the V1 subject split, window indices, full model
architecture, and trained full-model checkpoint unchanged.

## 1. Train the CGM-only baseline

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
.venv/bin/python -m src.train_cgm_only \
  --output-dir outputs/v1_cgm_only
```

The model sees historical CGM and Time only. Its future rollout sees Time only;
Insulin and Carb are ignored by construction.

## 2. Run all baseline and sensitivity evaluations

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
.venv/bin/python -m src.evaluate_action_sensitivity
```

Run the same experiments concurrently across the available GPUs, wait for all
children without polling, and merge automatically:

```bash
bash scripts/run_parallel_sensitivity.sh
```

The parallel launcher uses the currently idle GPUs 0, 1, 6, and 7 with batch
size 8192 and no DataLoader workers. Lightweight shards may share a GPU; this
avoids interfering with busy GPUs while limiting CPU/I/O amplification.

Outputs below `outputs/action_sensitivity/`:

- `unified_results.csv`: model/condition × event group × horizon table.
- `detailed_results.json`: micro, per-subject, subject-macro, group counts,
  and automatic action-dependence analysis.
- `analysis.json`: compact comparisons and warnings.

The default obvious Insulin event threshold is 0.5 U/5 min. Shuffle uses one
global, seeded permutation of complete future action sequences across the test
set, preserving their empirical marginal distribution while breaking their
window/state correspondence.
