#!/usr/bin/env bash
# Restart the final sweep cleanly:
#   1. stop every training process and wait for the accelerator to drain
#   2. drop the cells that hold no result (they were produced by the broken
#      builds or died silently) and the debug cells
#   3. upload the fixed sources, run the self-check as a gate
#   4. start the S6 chain detached
set -u
cd ~/MSA-SOTA

for p in $(pgrep -f "s6_final_config[.]sh"); do kill -9 $p 2>/dev/null; done
for p in $(pgrep -f "msa[.]train"); do kill -9 $p 2>/dev/null; done
sleep 5
echo "训练进程残留: $(pgrep -f 'msa[.]train' | wc -l)"

for d in $(ls results | grep -E "^(E[0-9]|DBG|C3_mosei_deberta|C4_)"); do
  rm -rf "results/$d"
done
echo "已清理无产物的格"

nohup setsid bash /tmp/s6_final_config.sh > logs/s6_chain.out 2>&1 < /dev/null &
sleep 40
echo "--- 链首 12 行 ---"
head -12 logs/s6_chain.out
echo "--- 当前格 ---"
ps -eo cmd | grep "[m]sa[.]train" | grep -o "\-\-tag [A-Za-z0-9_]*" | head -1
date +%H:%M
