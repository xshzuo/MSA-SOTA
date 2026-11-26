#!/usr/bin/env bash
# The noise floor: the same configuration and the same seed, run three times.
#
# Two runs of the same seed do not agree exactly (the non-deterministic GPU
# kernels move the result by about 1.7 pp Acc-7), so every decision line has
# to be read against this spread, not against a single run.
#
# The deterministic arm answers the other question: how much of the spread is
# the implementation and how much is the algorithm's own variance.
set -u
export PYTHONPATH=$HOME/MSA-SOTA/pylibs
export HF_HOME=$HOME/MSA-SOTA/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
cd $HOME/MSA-SOTA/code

DATA=$HOME/MSA-SOTA/data/mosi
RES=$HOME/MSA-SOTA/results

run () {   # tag, seed, extra args...
  local tag=$1; shift
  local seed=$1; shift
  local cell=$RES/$tag
  if ls "$cell" 2>/dev/null | grep -q "_result.csv"; then
    echo "--- 跳过已存在的 $tag"
    return 0
  fi
  rm -rf "$cell"; mkdir -p "$cell"
  echo "=== $tag (seed=$seed) $(date -Is) ==="
  /usr/bin/python3 -m msa.train --data $DATA --corpus mosi \
      --tokenizer roberta-large --cache_dir $HOME/MSA-SOTA/hf_cache \
      --seed "$seed" --out "$cell" --tag "$tag" \
      --llrd 0.9 --warmup_steps 200 --epochs 20 --max_len 64 --case_rule upper_words \
      "$@" 2>&1 | tee $HOME/MSA-SOTA/logs/$tag.log | \
      grep -E "^\[best|^\[token|Traceback|Error"
}

# the same seed three times, non-deterministic kernels
run N1_mosi_roberta_cnUW_s1111_rep1 1111
run N2_mosi_roberta_cnUW_s1111_rep2 1111
run N3_mosi_roberta_cnUW_s1111_rep3 1111
# the same seed twice, deterministic kernels: the two runs must agree exactly
run D1_mosi_roberta_cnUW_s1111_det 1111 --deterministic 1
run D2_mosi_roberta_cnUW_s1111_det 1111 --deterministic 1
echo "=== NOISE FLOOR DONE $(date -Is) ==="
