#!/usr/bin/env bash
# The follow-ups that can still change a conclusion, in priority order:
#   E10b (one seed)  a re-run of E10 that also saves the development
#                    predictions, so that the scale calibration of L5 can be
#                    fitted on the development split alone (one seed is enough
#                    to see whether the fitted scale transfers).
#   E12 (one seed)   the same recipe with an MSE loss: L1 fits the conditional
#                    median and compresses the output scale, MSE fits the mean.
#                    One seed decides whether the arm is worth three.
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

bash $HOME/MSA-SOTA/scripts/gpu_guard.sh s9 || exit 1

run E10b_mosi_deberta_bhead_ctxreg_s1111 1111 microsoft/deberta-v3-large mosi --epochs 20 \
    --binary_head 1 --lambda_binary 0.1 --context 1 --context_route reg
run E12_mosi_bhead_ctxreg_mse_s1111 1111 microsoft/deberta-v3-large mosi --epochs 20 \
    --binary_head 1 --lambda_binary 0.1 --context 1 --context_route reg --loss mse

echo "=== S9 DONE $(date -Is) ==="
