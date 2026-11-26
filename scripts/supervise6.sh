#!/usr/bin/env bash
# Wait for the S13 chain (which ends with the MOSEI deberta cell) and then
# run the S14 sweep that tests whether the two published axes can be held
# at once.
set -u
cd ~/MSA-SOTA
for i in $(seq 1 400); do
  if grep -q "=== S13 DONE" logs/s13_chain.out 2>/dev/null; then break; fi
  if ! pgrep -f "s13_tail[.]sh" > /dev/null 2>&1; then
    sleep 30
    if ! pgrep -f "s13_tail[.]sh" > /dev/null 2>&1; then
      echo "[sup6] S13 已结束（链退出）"
      break
    fi
  fi
  sleep 30
done
echo "[sup6] 启动 S14 $(date -Is)"
nohup setsid bash /tmp/s14_break_tradeoff.sh > logs/s14_chain.out 2>&1 < /dev/null &
sleep 25
head -3 logs/s14_chain.out 2>/dev/null
date +%H:%M
