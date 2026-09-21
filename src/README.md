# V1 continuous-state diabetes world model

The V1 model estimates a latent physiological state from two hours of observed
history, then recursively rolls that state forward using only future controls
and disturbances. True future CGM is never an input to the transition module.

## Inputs and rollout

Historical input has five channels:

```text
[CGM, Insulin, Carb, time_sin, time_cos]
```

The main model encodes CGM, Insulin, Carb, and Time separately. Insulin and Carb
carry learnable concept embeddings. Their representations are fused into an
event embedding, and a GRU History Encoder maps 24 past steps to the current
latent state. The future transition receives four channels only:

```text
[Insulin, Carb, time_sin, time_cos]
```

A single shared GRUCell recursively transitions the latent state for 12 future
steps. An MLP decodes CGM from every state. Metrics are reported for steps 1,
6, and 12, corresponding to 5, 30, and 60 minutes.

The `baseline` model sends the five historical channels directly into a GRU
and uses a simple control-only GRUCell rollout, without modality encoders or
concept embeddings.

## Leakage controls

- Existing subject-level train/validation/test manifests are consumed as-is.
- Normalization statistics are computed from train subjects only.
- CGM uses train-set mean/std z-score normalization. Insulin and Carb use
  `log1p(x)` followed by train-set mean/std z-score normalization. Validation
  and test reuse these exact train statistics without refitting.
- Every window is checked to remain inside one `segment_id`.
- Every history and target row must have both `cgm_observed=True` and
  `insulin_observed=True`.
- Future CGM is returned only as the target and cannot be passed into either
  model's transition API.
- Subject/source identifiers are returned as evaluation metadata only and are
  never passed to model inputs.

`test_metrics.json` stores window-micro metrics, separate per-subject metrics,
and an equal-subject-weight macro average for MAE, RMSE, and MARD at 5, 30, and
60 minutes (plus the overall 12-step rollout summary).

## Train

Main model:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
.venv/bin/python -m src.train \
  --model world_model \
  --output-dir outputs/v1_world_model
```

The GPU-tested defaults are `batch_size=2048` and `num_workers=4`. On the
current training server this configuration uses only about 1.14 GiB on the RTX
5880. Its measured epoch time is within roughly 3% of an A800 for this small
recurrent model, so the RTX 5880 is the resource-efficient default.

Baseline:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
.venv/bin/python -m src.train \
  --model baseline \
  --output-dir outputs/v1_baseline
```

For a CPU smoke test:

```bash
/home/wanghaobo/.conda/envs/pt110/bin/python -m src.train \
  --epochs 1 --batch-size 32 --hidden-dim 32 --embedding-dim 8 --event-dim 16 \
  --max-train-windows 256 --max-val-windows 128 --max-test-windows 128 \
  --output-dir /tmp/diabetes_world_model_smoke
```

Outputs include `best.pt`, `history.jsonl`, and `test_metrics.json`.
