#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/gaurav/newdata"
LOGDIR="$ROOT/reports/rtdetr_overnight_logs"
mkdir -p "$LOGDIR"

export PYTORCH_ENABLE_MPS_FALLBACK=1

echo "===== RT-DETR overnight run started: $(date) ====="
echo "Logs: $LOGDIR"

echo
echo "===== 1. Training RT-DETR original-only ====="
python3 "$ROOT/src/non_yolo/train_rtdetr_original_only.py" 2>&1 | tee "$LOGDIR/01_rtdetr_original_only.log"

echo
echo "Finding latest original-only best.pt..."
LATEST_ORIG=$(find "$ROOT/runs/rtdetr_original_only" -path "*/weights/best.pt" -type f -print0 | xargs -0 ls -t | head -1)

if [ -z "$LATEST_ORIG" ]; then
  echo "ERROR: No original-only best.pt found."
  exit 1
fi

echo "Latest original-only weights: $LATEST_ORIG"

echo
echo "Patching curriculum script to use latest original-only weights..."
python3 - << EOF_PY
from pathlib import Path

script = Path("$ROOT/src/non_yolo/train_rtdetr_stagebsoft_curriculum_stage2.py")
txt = script.read_text()

lines = []
for line in txt.splitlines():
    if line.strip().startswith("INIT_WEIGHTS ="):
        lines.append('INIT_WEIGHTS = Path("' + "$LATEST_ORIG" + '")')
    else:
        lines.append(line)

script.write_text("\\n".join(lines) + "\\n")
print("Updated:", script)
print("INIT_WEIGHTS:", "$LATEST_ORIG")
EOF_PY

echo
echo "===== 2. Training RT-DETR Stage B-soft direct ====="
python3 "$ROOT/src/non_yolo/train_rtdetr_stagebsoft_direct.py" 2>&1 | tee "$LOGDIR/02_rtdetr_stagebsoft_direct.log"

echo
echo "===== 3. Training RT-DETR Stage B-soft curriculum ====="
python3 "$ROOT/src/non_yolo/train_rtdetr_stagebsoft_curriculum_stage2.py" 2>&1 | tee "$LOGDIR/03_rtdetr_stagebsoft_curriculum.log"

echo
echo "===== 4. Evaluating RT-DETR models ====="
python3 "$ROOT/src/non_yolo/eval_rtdetr_generalization.py" 2>&1 | tee "$LOGDIR/04_rtdetr_eval.log"

echo
echo "===== RT-DETR overnight run completed: $(date) ====="
echo "Result table:"
echo "$ROOT/reports/rtdetr_generalization_current/rtdetr_generalization_results.md"

if [ -f "$ROOT/reports/rtdetr_generalization_current/rtdetr_generalization_results.md" ]; then
  cat "$ROOT/reports/rtdetr_generalization_current/rtdetr_generalization_results.md"
fi
