#!/usr/bin/env bash
# Quick progress report (read-only): bash scripts/remote/status.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$REPO"
echo "== session '$SESSION': $(tmux has-session -t "$SESSION" 2>/dev/null && echo running || echo 'not running')"
echo "== status: $(cat "$LOGS/STATUS" 2>/dev/null || echo 'not started')"
echo "== finished runs (test, complete pairs):"
"$VENV/bin/python" - <<'PY'
import json, glob
rows = sorted(glob.glob("outputs/runs/upmc/*/seed*/eval/test_metrics.json"))
for f in rows:
    m = json.load(open(f)); p = f.split("/")
    print(f"   {p[3]:16s} {p[4]:9s} acc {100*m['accuracy']:6.2f}  macro-F1 {100*m['macro_f1']:6.2f}")
print(f"   ({len(rows)} finished)")
PY
cur=$(ls -t outputs/runs/upmc/*/seed*/train.log 2>/dev/null | head -1)
[ -n "$cur" ] && { echo "== latest run log: $cur"; grep -E "epoch|Best|test\]|Early|Error|error" "$cur" | tail -6; }
echo "== master log (last 8 lines):"; tail -n 8 "$LOGS/m2_latest.log" 2>/dev/null
echo "== GPUs:"; nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader
echo "== disk /home: $(df -h "$A3_WORK" | awk 'NR==2{print $4" free"}')"
