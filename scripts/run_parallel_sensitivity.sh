#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

RUN_ID="${1:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="outputs/action_sensitivity_parallel/${RUN_ID}"
mkdir -p "$RUN_ROOT"
echo "$RUN_ROOT" > outputs/action_sensitivity_parallel/latest_run.txt

declare -a PIDS=()
declare -a NAMES=()
declare -a DIRS=()

launch() {
  local gpu="$1"
  local name="$2"
  shift 2
  local output_dir="$RUN_ROOT/$name"
  mkdir -p "$output_dir"
  echo "launching $name on physical GPU $gpu: $*"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" \
    .venv/bin/python -u -m src.evaluate_action_sensitivity \
      --batch-size 8192 \
      --num-workers 0 \
      --experiments "$@" \
      --output-dir "$output_dir" \
      >"$output_dir/run.log" 2>&1 &
  PIDS+=("$!")
  NAMES+=("$name")
  DIRS+=("$output_dir")
}

launch 0 full_persistence full persistence
launch 1 zero_insulin_cgm zero_insulin cgm_only
launch 6 zero_carb zero_carb
launch 7 zero_both zero_both
launch 6 shuffle_insulin shuffle_insulin
launch 1 shuffle_carb shuffle_carb

failed=0
for index in "${!PIDS[@]}"; do
  if wait "${PIDS[$index]}"; then
    echo "completed ${NAMES[$index]}"
  else
    echo "FAILED ${NAMES[$index]} (see ${DIRS[$index]}/run.log)" >&2
    failed=1
  fi
done

if [[ "$failed" -ne 0 ]]; then
  exit 1
fi

.venv/bin/python -m src.merge_sensitivity_results \
  --input-dirs "${DIRS[@]}" \
  --output-dir "$RUN_ROOT/final" \
  >"$RUN_ROOT/merge.log" 2>&1

echo "all experiments complete: $RUN_ROOT/final"
