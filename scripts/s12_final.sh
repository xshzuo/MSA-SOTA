#!/usr/bin/env bash
# The final sweep: complete the arm that clears four of the five columns and
# test the one remaining combination that has not been tried.
#
#   E14  deberta + the joint decision head with lambda_binary 0.5, no context.
#        Its first seed reads Acc-2 87.76 / 89.79, Acc-7 50.00, MAE 0.5878,
#        Corr 0.8690 -- four of the five acceptance columns.  One seed is not
#        a result, so the two missing seeds are the priority.
#   E17  the same arm with an MSE loss.  The two interventions that moved
#        Acc-2 upwards (the MSE loss and the larger decision-head weight) were
#        each measured with the context present; their combination with each
#        other has not.
set -u
export PYTHONPATH=$HOME/MSA-SOTA/pylibs
export HF_HOME=$HOME/MSA-SOTA/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
cd $HOME/MSA-SOTA/code

RES=$HOME/MSA-SOTA/results
LOGS=$HOME/MSA-SOTA/logs

run () {   # tag, seed, tokenizer, corpus, extra args...
  local tag=$1; shift
  local seed=$1; shift
  local tok=$1; shift
  local corpus=$1; shift
  local cell=$RES/$tag
  if ls "$cell" 2>/dev/null | grep -q "_result.csv"; then
    echo "--- 跳过已存在的 $tag"; return 0
  fi
  rm -rf "$cell"; mkdir -p "$cell"
  echo "=== $tag (seed=$seed, $tok, $corpus) $(date -Is) ==="
  local extra=""
  if [ "$tok" = "microsoft/deberta-v3-large" ]; then extra="--batch_size 8 --accum 4"; fi
  /usr/bin/python3 -m msa.train --data $HOME/MSA-SOTA/data/$corpus --corpus $corpus \
      --tokenizer "$tok" --cache_dir $HOME/MSA-SOTA/hf_cache \
      --seed "$seed" --out "$cell" --tag "$tag" \
      --llrd 0.9 --warmup_steps 200 --max_len 64 --case_rule upper_words $extra "$@" \
      > "$LOGS/$tag.log" 2>&1
  local rc=$?
  grep -E "^\[best|^\[token|Traceback|Error" "$LOGS/$tag.log"
  echo "[EXIT_STATUS] $tag rc=$rc $(date -Is)"
  if [ "$rc" != "0" ]; then echo "!!! $tag 失败 rc=$rc，该格作废"; fi
}

bash $HOME/MSA-SOTA/scripts/gpu_guard.sh s12 || exit 1

run E14_mosi_deberta_bhead_lb05_s2222 2222 microsoft/deberta-v3-large mosi --epochs 20 \
    --binary_head 1 --lambda_binary 0.5
run E14_mosi_deberta_bhead_lb05_s3333 3333 microsoft/deberta-v3-large mosi --epochs 20 \
    --binary_head 1 --lambda_binary 0.5
run E17_mosi_deberta_lb05_mse_s1111 1111 microsoft/deberta-v3-large mosi --epochs 20 \
    --binary_head 1 --lambda_binary 0.5 --loss mse
run E17_mosi_deberta_lb05_mse_s2222 2222 microsoft/deberta-v3-large mosi --epochs 20 \
    --binary_head 1 --lambda_binary 0.5 --loss mse
run E17_mosi_deberta_lb05_mse_s3333 3333 microsoft/deberta-v3-large mosi --epochs 20 \
    --binary_head 1 --lambda_binary 0.5 --loss mse
run E14_mosi_deberta_bhead_lb05_s4444 4444 microsoft/deberta-v3-large mosi --epochs 20 \
    --binary_head 1 --lambda_binary 0.5
run E14_mosi_deberta_bhead_lb05_s5555 5555 microsoft/deberta-v3-large mosi --epochs 20 \
    --binary_head 1 --lambda_binary 0.5

echo "=== S12 DONE $(date -Is) ==="
