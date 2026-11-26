#!/usr/bin/env bash
# Wait for the S11 chain to finish, then start the S12 final sweep.
set -u
cd ~/MSA-SOTA
for i in $(seq 1 200); do
  if grep -q "=== S11 DONE" logs/s11_chain.out 2>/dev/null; then break; fi
  if ! pgrep -f "s11_last[.]sh" > /dev/null 2>&1; then
    sleep 20
    if ! pgrep -f "s11_last[.]sh" > /dev/null 2>&1; then
      echo "[sup3] S11 已结束，继续 S12"
      break
    fi
  fi
  sleep 30
done
echo "[sup3] 启动 S12 $(date -Is)"
nohup setsid bash /tmp/s12_final.sh > logs/s12_chain.out 2>&1 < /dev/null &
sleep 25
head -4 logs/s12_chain.out
date +%H:%M
