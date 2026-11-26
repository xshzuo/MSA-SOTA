#!/usr/bin/env bash
# Chain supervisor.
#
#   1. wait until the three E10 cells are finished
#   2. stop the S8 chain there, so that the six MOSEI-DeBERTa cells (about
#      three hours) do not start before the arms that can still change a
#      conclusion have been run
#   3. the S9 queue (already waiting) takes over; when it is done, start S10
set -u
cd ~/MSA-SOTA

echo "[sup] 等待 E10 三格完成"
while pgrep -f "E10_mosi_deberta_bhead_ctxreg" > /dev/null; do sleep 30; done
echo "[sup] E10 完成，停掉 S8 链 $(date -Is)"

for p in $(pgrep -f "s8_large_encoder[.]sh"); do kill -9 $p 2>/dev/null; done
for p in $(pgrep -f "msa[.]train"); do kill -9 $p 2>/dev/null; done
sleep 5
rm -rf results/C3_mosei_deberta_cpf_cnUW_s* results/C4_mosei_deberta_bhead_s*
echo "[sup] S8 已停，训练进程 $(pgrep -f 'msa[.]train' | wc -l)"

# the S9 queue picks the chain up on its own; wait for it to finish
echo "[sup] 等待 S9"
for i in $(seq 1 120); do
  if grep -q "=== S9 DONE" logs/s9_calibration_and_loss_chain.out 2>/dev/null; then break; fi
  if ! pgrep -f "s9_calibration_and_loss" > /dev/null 2>&1; then
    sleep 20
    if ! pgrep -f "s9_calibration_and_loss" > /dev/null 2>&1; then
      echo "[sup] S9 未在运行（可能已结束或未启动），继续"
      break
    fi
  fi
  sleep 30
done
echo "[sup] 启动 S10 $(date -Is)"
nohup setsid bash /tmp/s10_endgame.sh > logs/s10_chain.out 2>&1 < /dev/null &
sleep 20
head -5 logs/s10_chain.out
date +%H:%M
