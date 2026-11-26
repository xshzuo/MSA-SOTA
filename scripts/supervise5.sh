#!/usr/bin/env bash
# Wait for the S12 chain to finish, then run the tail cells (S13).
set -u
cd ~/MSA-SOTA
for i in $(seq 1 600); do
  if grep -q "=== S12 DONE" logs/s12_chain.out 2>/dev/null; then break; fi
  if ! pgrep -f "s12_final[.]sh" > /dev/null 2>&1; then
    sleep 30
    if ! pgrep -f "s12_final[.]sh" > /dev/null 2>&1; then
      echo "[sup5] S12 已结束（链退出）"
      break
    fi
  fi
  sleep 30
done
echo "[sup5] 启动 S13 $(date -Is)"
nohup setsid bash /tmp/s13_tail.sh > logs/s13_chain.out 2>&1 < /dev/null &
sleep 20
head -3 logs/s13_chain.out 2>/dev/null
date +%H:%M
