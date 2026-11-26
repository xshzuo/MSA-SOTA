#!/usr/bin/env bash
# Queue a follow-up sweep behind the running chain.
#
# Usage: bash scripts/queue_next.sh <script-to-run> <chain-log-to-watch>
set -u
NEXT=$1
WATCH=${2:-s8_chain}
cd ~/MSA-SOTA

for i in $(seq 1 300); do
  if grep -qE "=== S[0-9] DONE" "logs/$WATCH.out" 2>/dev/null; then break; fi
  if ! pgrep -f "${WATCH%.out}_" > /dev/null 2>&1 && ! pgrep -f "s8_large_encoder" > /dev/null 2>&1 \
     && ! pgrep -f "s9_" > /dev/null 2>&1; then
    sleep 20
    if ! pgrep -f "s8_large_encoder" > /dev/null 2>&1; then
      echo "前一条链已结束，继续排队 $NEXT"
      break
    fi
  fi
  sleep 30
done
echo "启动 $NEXT $(date -Is)"
nohup setsid bash "/tmp/$NEXT" > "logs/${NEXT%.sh}_chain.out" 2>&1 < /dev/null &
sleep 30
head -5 "logs/${NEXT%.sh}_chain.out"
date +%H:%M
