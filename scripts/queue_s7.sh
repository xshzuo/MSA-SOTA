#!/usr/bin/env bash
# Queue the follow-up sweep behind the running S6 chain.
set -u
cd ~/MSA-SOTA

# wait for the S6 chain to finish (or die) before touching the accelerator
for i in $(seq 1 240); do
  if grep -q "=== S6 DONE" logs/s6_chain.out 2>/dev/null; then break; fi
  if ! pgrep -f "s6_final_config[.]sh" > /dev/null; then
    sleep 20
    if ! pgrep -f "s6_final_config[.]sh" > /dev/null; then
      echo "S6 链已不在运行（正常结束或异常退出），继续排队 S7"
      break
    fi
  fi
  sleep 30
done
echo "S7 开始 $(date -Is)"
nohup setsid bash /tmp/s7_followup.sh > logs/s7_chain.out 2>&1 < /dev/null &
sleep 30
head -5 logs/s7_chain.out
date +%H:%M
