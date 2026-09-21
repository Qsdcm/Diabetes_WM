# Diabetes world-model preprocessing

`preprocess_world_model.py` converts AZT1D and BrisT1D-Open into a shared
5-minute representation for world-model training.

## Canonical features

Each output row contains these model features:

- `cgm_mg_dl`: CGM in mg/dL.
- `insulin_u_5min`: total insulin delivered in the bin, in U/5 min.
- `carb_g_5min`: carbohydrate events in g/5 min.
- `time_sin`, `time_cos`: local time-of-day encoded on a 24-hour cycle.

`timestamp`, `dataset`, `subject_id`, and `segment_id` identify each continuous
sequence. `cgm_observed` and `insulin_observed` distinguish measured values from
short-gap fills and should normally be retained for auditing, even if they are
not passed to the model.

## Dataset-specific mapping

| Canonical field | AZT1D | BrisT1D-Open |
|---|---|---|
| CGM | `CGM` (Subject 14: `Readings (CGM / BGM)`), already mg/dL | `bg` in mmol/L, multiplied by 18.0182 |
| Insulin | `TotalBolusInsulinDelivered + Basal / 12` | `insulin`, already U/5 min |
| Carb | `CarbSize` | `carbs` |
| Time | `EventDateTime` | `timestamp` |

AZT1D has OCR-derived basal rates with dropped decimal points (for example,
`825` beside `0.825`). Values above 10 U/h are divided by 1000 before the
U/h-to-U/5 min conversion. Repeated merge rows are collapsed before events are
aggregated, preventing duplicate bolus and carbohydrate doses.

CGM gaps up to 30 minutes are linearly interpolated. AZT1D basal rates are
forward-filled for at most 30 minutes. Longer gaps are not replaced with zero;
they split the output into separate `segment_id` values.

Negative insulin or carbohydrate values are treated as invalid source values,
not as physiological inputs. Subject P17 has no insulin observations in the
open BrisT1D data and is therefore reported but not emitted as a model sequence.

## Run

```bash
/home/wanghaobo/.conda/envs/pt110/bin/python preprocessing/preprocess_world_model.py
```

Default output:

```text
./Dataset_5min/
├── AZT1D/*.csv
├── BrisT1D-Open/*.csv
├── all_sequences.csv
└── preprocessing_report.json
```

Run the unit tests with:

```bash
/home/wanghaobo/.conda/envs/pt110/bin/python -m unittest -v \
  preprocessing.test_preprocess_world_model
```

## Leakage-safe V1 windows

Build subject-disjoint train/validation/test indices only after the unified
subject files exist:

```bash
/home/wanghaobo/.conda/envs/pt110/bin/python \
  preprocessing/build_window_index.py
```

The V1 defaults use 24 history steps (2 hours), 12 target steps (1 hour), and
stride 1. The split is deterministic and stratified by source dataset. A
subject can occur in exactly one split. Windows never cross `segment_id`, and
every input and target row must satisfy both `cgm_observed=True` and
`insulin_observed=True`; interpolated CGM is therefore never used as a target.

The generated files are written below `Dataset_5min/window_index_v1/`:

```text
train_windows.csv
val_windows.csv
test_windows.csv
split_manifest.json
```
