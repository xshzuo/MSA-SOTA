#!/usr/bin/env bash
# Can the trade-off be broken?
#
# Every intervention we tried moved Acc-2 and Acc-7 in opposite directions.
# The three that raised Acc-2 were: the MSE loss, a larger decision-head
# weight, and the routed context.  The three that raised Acc-7 were: the
# context, the L1 loss, and a smaller decision-head weight.  E18 combines the
# Acc-2-favouring direction (MSE + lambda_binary 0.5) with the Acc-7-favouring
# one (the routed context) in a single arm.  If the trade-off is a shared
# degree of freedom, this arm lands on the frontier again; if it is two
# separate effects, the arm can hold both.
set -u
cd ~/MSA-SOTA
export PYTHONPATH=$HOME/MSA-SOTA/pylibs
export HF_HOME=$HOME/MSA-SOTA/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0

run_cell () {
  local tag=$1; shift
  local seed=$1; shift
  local cell=$HOME/MSA-SOTA/results/$tag
  if ls "$cell" 2>/dev/null | grep -q "_result.csv"; then
    echo "--- 跳过已存在的 $tag"; return 0
  fi
  for attempt in $(seq 1 20); do
    if bash $HOME/MSA-SOTA/scripts/gpu_guard.sh "$tag"; then break; fi
    echo "[retry] $tag 闸门拒绝，等待后重试 (第 $attempt 次)"
    sleep 120
  done
  rm -rf "$cell"; mkdir -p "$cell"
  echo "=== $tag (seed=$seed) $(date -Is) ==="
  cd $HOME/MSA-SOTA/code
  /usr/bin/python3 -m msa.train --data $HOME/MSA-SOTA/data/mosi --corpus mosi \
      --tokenizer microsoft/deberta-v3-large --cache_dir $HOME/MSA-SOTA/hf_cache \
      --seed "$seed" --out "$cell" --tag "$tag" --llrd 0.9 --warmup_steps 200 \
      --max_len 64 --case_rule upper_words --batch_size 8 --accum 4 --epochs 20 \
      --binary_head 1 --lambda_binary 0.5 --context 1 --context_route reg --loss mse \
      > $HOME/MSA-SOTA/logs/$tag.log 2>&1
  local rc=$?
  grep -E "^\[best|Traceback|Error" $HOME/MSA-SOTA/logs/$tag.log
  echo "[EXIT_STATUS] $tag rc=$rc $(date -Is)"
  while pgrep -f "msa[.]train" > /dev/null; do sleep 15; done
  sleep 10
}

for s in 1111 2222 3333; do
  run_cell E18_mosi_deberta_lb05_mse_ctxreg_s$s $s
done

# The headline routing arm goes to five seeds: with three seeds even the
# largest effect we measured (the routing effect) only reaches P = 0.082 in a
# paired t test, so the primary comparison needs more than three seeds.
for s in 4444 5555; do
  run_cell E8_mosi_bhead_ctxreg_s$s $s
done

echo "=== S14 DONE $(date -Is) ==="
