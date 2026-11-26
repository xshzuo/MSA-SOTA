#!/usr/bin/env bash
# S4 -- the context stream, gated by the same attributable mechanism.
#
#   D0  RoBERTa-large + CPF + the case front-end      (the best confirmed arm)
#   D1  D0 + the neighbouring utterance as a third stream
#   D2  D1 + the attributable gate over all three streams
#
# The neighbouring utterance is read with the *same* encoders, so the context
# costs no extra parameters; the gate decides per sample how much of the
# context to use, because a neighbouring utterance is not always informative.
set -u
export PYTHONPATH=$HOME/MSA-SOTA/pylibs
export HF_HOME=$HOME/MSA-SOTA/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
cd $HOME/MSA-SOTA/code

RES=$HOME/MSA-SOTA/results

run () {   # tag, seed, corpus, extra args...
  local tag=$1; shift
  local seed=$1; shift
  local corpus=$1; shift
  local cell=$RES/$tag
  if ls "$cell" 2>/dev/null | grep -q "_result.csv"; then
    echo "--- 跳过已存在的 $tag"
    return 0
  fi
  rm -rf "$cell"; mkdir -p "$cell"
  echo "=== $tag (seed=$seed) $(date -Is) ==="
  /usr/bin/python3 -m msa.train --data $HOME/MSA-SOTA/data/$corpus --corpus $corpus \
      --tokenizer roberta-large --cache_dir $HOME/MSA-SOTA/hf_cache \
      --seed "$seed" --out "$cell" --tag "$tag" \
      --llrd 0.9 --warmup_steps 200 --max_len 64 --case_rule upper_words "$@" 2>&1 | \
      tee $HOME/MSA-SOTA/logs/$tag.log | \
      grep -E "^\[data\]|^\[best|^\[token|Traceback|Error"
}

for s in 1111 2222 3333; do
  run D1_mosi_ctx_s$s $s mosi --epochs 20 --context 1
done
for s in 1111 2222 3333; do
  run D2_mosi_ctx_asf_s$s $s mosi --epochs 20 --context 1 --asf 1 --asf_lambda 1.0
done
# the MOSEI context arm is only worth the compute once the MOSI context arm
# has shown a gain, so it lives in its own script (s4b_context_mosei.sh)
echo "=== S4 DONE $(date -Is) ==="
