#!/usr/bin/env bash
# Wait for the last E14 cell, then stop the S10 chain (its remaining cells are
# the two MOSEI arms, which the S11 script also runs) and start S11.
set -u
cd ~/MSA-SOTA

echo "[sup2] 等待 E14_s3333"
while pgrep -f "E14_mosi_deberta_bhead_lb05_s3333" > /dev/null; do sleep 30; done
sleep 10
for p in $(pgrep -f "s10_endgame[.]sh"); do kill -9 $p 2>/dev/null; done
for p in $(pgrep -f "msa[.]train"); do kill -9 $p 2>/dev/null; done
sleep 5
rm -rf results/C4_mosei_deberta_bhead_s* results/C5_mosei_bhead_ctxreg_s*
echo "[sup2] S10 已停，启动 S11 $(date -Is)"
nohup setsid bash /tmp/s11_last.sh > logs/s11_chain.out 2>&1 < /dev/null &
sleep 25
head -4 logs/s11_chain.out
date +%H:%M
