#!/usr/bin/env bash
# Priorities: the headline arm needs several seeds, the last MOSEI cell does
# not.  So: let the MOSEI routing cell finish, stop the S11 chain (which
# would otherwise spend half an hour on the MOSEI deberta cell), let S12 run,
# and only then run the MOSEI deberta cell.
set -u
cd ~/MSA-SOTA

echo "[sup4] 等待 C5_s3333"
while pgrep -f "C5_mosei_bhead_ctxreg_s3333" > /dev/null; do sleep 20; done
sleep 10
for p in $(pgrep -f "s11_last[.]sh"); do kill -9 $p 2>/dev/null; done
sleep 5
rm -rf results/C4_mosei_deberta_bhead_s1111
echo "[sup4] S11 已停，S12 应接管 $(date -Is)"

# wait for S12 to finish
for i in $(seq 1 400); do
  if grep -q "=== S12 DONE" logs/s12_chain.out 2>/dev/null; then break; fi
  if ! pgrep -f "s12_final[.]sh" > /dev/null 2>&1; then sleep 30; \
     if ! pgrep -f "s12_final[.]sh" > /dev/null 2>&1; then break; fi; fi
  sleep 30
done
echo "[sup4] S12 已结束，启动 C4 $(date -Is)"
mkdir -p results/C4_mosei_deberta_bhead_s1111
export PYTHONPATH=$HOME/MSA-SOTA/pylibs HF_HOME=$HOME/MSA-SOTA/hf_cache
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
cd $HOME/MSA-SOTA/code
/usr/bin/python3 -m msa.train --data $HOME/MSA-SOTA/data/mosei --corpus mosei \
  --tokenizer microsoft/deberta-v3-large --cache_dir $HOME/MSA-SOTA/hf_cache \
  --seed 1111 --out $HOME/MSA-SOTA/results/C4_mosei_deberta_bhead_s1111 \
  --tag C4_mosei_deberta_bhead_s1111 --llrd 0.9 --warmup_steps 200 --max_len 64 \
  --case_rule upper_words --batch_size 8 --accum 4 --epochs 20 \
  --binary_head 1 --lambda_binary 0.1 > $HOME/MSA-SOTA/logs/C4_mosei_deberta_bhead_s1111.log 2>&1
echo "[EXIT_STATUS] C4 rc=$? $(date -Is)"
grep -E "^\[best" $HOME/MSA-SOTA/logs/C4_mosei_deberta_bhead_s1111.log
echo "=== ALL DONE $(date -Is) ==="
