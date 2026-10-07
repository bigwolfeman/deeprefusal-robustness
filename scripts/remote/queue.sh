#!/usr/bin/env bash
# Sequential GPU job queue for the 3070. Jobs file: one `name<TAB>command` per line.
# The runner pops the first line each time, so jobs can be appended or reordered while it runs.
# Status: $ROOT/logs/queue/status.jsonl. Per-job logs: $ROOT/logs/queue/<name>.log
set -uo pipefail
ROOT=/mnt/bigdata/deeprefusal
JOBS=$ROOT/queue/jobs.tsv
LOGS=$ROOT/logs/queue
mkdir -p "$LOGS" "$ROOT/queue"
. "$ROOT/env.sh"
cd "$ROOT/repo"
while true; do
  line=$(flock "$JOBS" bash -c "head -n1 '$JOBS'; sed -i '1d' '$JOBS'")
  [ -z "$line" ] && { echo "{\"event\":\"queue_empty\",\"t\":\"$(date -Is)\"}" >> "$LOGS/status.jsonl"; break; }
  name=${line%%$'\t'*}; cmd=${line#*$'\t'}
  echo "{\"event\":\"start\",\"job\":\"$name\",\"t\":\"$(date -Is)\",\"sha\":\"$(git rev-parse --short HEAD)\"}" >> "$LOGS/status.jsonl"
  bash -c "$cmd" > "$LOGS/$name.log" 2>&1
  rc=$?
  echo "{\"event\":\"end\",\"job\":\"$name\",\"rc\":$rc,\"t\":\"$(date -Is)\"}" >> "$LOGS/status.jsonl"
  [ $rc -ne 0 ] && echo "[queue] JOB FAILED: $name rc=$rc (see $LOGS/$name.log)" >&2
done
