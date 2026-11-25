#!/usr/bin/env bash
# Start run_m2.sh in a detached tmux session, so it keeps running when the AnyDesk window / terminal is closed.
#   bash scripts/remote/launch.sh            (optional: A3_GPU=1 A3_WORKERS=16 A3_SEEDS="42" bash scripts/remote/launch.sh)
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session '$SESSION' is already running. Watch it: tmux attach -t $SESSION   (leave: Ctrl-b then d)"; exit 1
fi
# the tmux server is shared by everyone on this account: pass our settings explicitly
vars="A3_WORK=$A3_WORK"
for v in A3_GPU A3_WORKERS A3_SEEDS A3_MODELS A3_RETRIES A3_MIN_FREE_GB A3_MAX_WAIT_H; do
  [ -n "${!v:-}" ] && vars="$vars $v='${!v}'"
done
tmux new-session -d -s "$SESSION" "env $vars bash '$REPO/scripts/remote/run_m2.sh'; echo; echo '--- finished, press Enter to close ---'; read"
echo "Started tmux session '$SESSION'."
echo "  progress:  bash scripts/remote/status.sh"
echo "  live view: tmux attach -t $SESSION      (leave without stopping: Ctrl-b then d)"
echo "  stop:      tmux kill-session -t $SESSION (re-launch later: finished runs are kept, others resume)"
