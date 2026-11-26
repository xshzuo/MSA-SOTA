#!/usr/bin/env bash
# S3 -- the same three decision lines on CMU-MOSEI.
#
#   C0  RoBERTa-large + CPF, no case front-end
#   C1  C0 + the case front-end (upper_words)
#   C2  C1 + the attributable selective fusion
#   C3  C1 with DeBERTa-v3-large            (encoder-family robustness)
#
# The MOSEI transcripts are ordinary sentences, so the case front-end is
# expected to be *inert* here: it must not hurt.  That is the other half of
# the L1 claim, and it is the half that the corpus-aware decision rule buys.
set -u
export PYTHONPATH=$HOME/MSA-SOTA/pylibs
export HF_HOME=$HOME/MSA-SOTA/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
cd $HOME/MSA-SOTA/code

DATA=$HOME/MSA-SOTA/data/mosei
RES=$HOME/MSA-SOTA/results
mkdir -p $RES

run () {   # tag, seed, tokenizer, extra args...
  local tag=$1; shift
  local seed=$1; shift
  local tk=$1; shift
  local cell=$RES/$tag
  if ls "$cell" 2>/dev/null | grep -q "_result.csv"; then
    echo "--- 跳过已存在的 $tag"
    return 0
  fi
  rm -rf "$cell"; mkdir -p "$cell"
  echo "=== $tag (seed=$seed) $(date -Is) ==="
  /usr/bin/python3 -m msa.train --data $DATA --corpus mosei \
      --tokenizer "$tk" --cache_dir $HOME/MSA-SOTA/hf_cache \
      --seed "$seed" --out "$cell" --tag "$tag" \
      --llrd 0.9 --warmup_steps 200 --epochs 12 --max_len 64 "$@" 2>&1 | \
      tee $HOME/MSA-SOTA/logs/$tag.log | \
      grep -E "^\[data\]|^\[best|^\[token|Traceback|Error"
}

SEEDS="1111 2222 3333"
for s in $SEEDS; do
  run C0_mosei_roberta_cpf_s$s $s roberta-large --case_rule none
done
for s in $SEEDS; do
  run C1_mosei_roberta_cpf_cnUW_s$s $s roberta-large --case_rule upper_words
done
for s in $SEEDS; do
  run C2_mosei_roberta_cpf_cnUW_asf_s$s $s roberta-large --case_rule upper_words \
      --asf 1 --asf_lambda 1.0
done
run C3_mosei_deberta_cpf_cnUW_s1111 1111 microsoft/deberta-v3-large --case_rule upper_words
echo "=== S3 DONE $(date -Is) ==="
