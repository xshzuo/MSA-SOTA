#!/usr/bin/env bash
# S2 -- the decision experiments on CMU-MOSI (docs/00_INNOVATION_PLAN.md).
#
#   A0  BERT-base, flat learning rate, no warmup        (the baseline)
#   A1  A0 + CPF (layer-wise LR decay + warmup)         -> decision line L2
#   B0  RoBERTa-large + CPF, no case front-end           (the cased encoder, damaged input)
#   B1  B0 + the case front-end (upper_words)            -> decision line L1
#   B2  B1 + the attributable selective fusion            -> decision line L3
#   P1  B1 but with DeBERTa-v3-large                     (encoder-family robustness)
#   P2  B1 but with XLNet + the case front-end           (a second cased family)
#
# Three seeds per arm, so the lines can be judged with a paired statistic.
# The cell directories are cleared before the run and every result file
# records its own configuration.
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
mkdir -p $RES

run () {   # tag, seed, tokenizer, extra args...
  local tag=$1; shift
  local seed=$1; shift
  local tk=$1; shift
  local cell=$RES/$tag
  # a cell that already has a result is kept: re-running it has to be explicit
  if ls "$cell" 2>/dev/null | grep -q "_result.csv"; then
    echo "--- 跳过已存在的 $tag"
    return 0
  fi
  rm -rf "$cell"; mkdir -p "$cell"
  echo "=== $tag (seed=$seed) $(date -Is) ==="
  /usr/bin/python3 -m msa.train --data $DATA --corpus mosi \
      --tokenizer "$tk" --cache_dir $HOME/MSA-SOTA/hf_cache \
      --seed "$seed" --out "$cell" --tag "$tag" "$@" 2>&1 | \
      tee $HOME/MSA-SOTA/logs/$tag.log | \
      grep -E "^\[model\]|^\[best|^\[token|Traceback|Error"
}

SEEDS="1111 2222 3333"

# --- L2: the fine-tuning recipe (cheap arms, BERT-base)
for s in $SEEDS; do
  run A0_mosi_bert_flat_s$s $s bert-base-uncased --llrd 1.0 --warmup_steps 0 --epochs 20
done
for s in $SEEDS; do
  run A1_mosi_bert_cpf_s$s $s bert-base-uncased --llrd 0.9 --warmup_steps 200 --epochs 20
done

# --- L1: the case front-end on a case-sensitive encoder
for s in $SEEDS; do
  run B0_mosi_roberta_cpf_s$s $s roberta-large --llrd 0.9 --warmup_steps 200 --epochs 20 \
      --max_len 64 --case_rule none
done
for s in $SEEDS; do
  run B1_mosi_roberta_cpf_cnUW_s$s $s roberta-large --llrd 0.9 --warmup_steps 200 --epochs 20 \
      --max_len 64 --case_rule upper_words
done

# --- L3: the attributable selective fusion
for s in $SEEDS; do
  run B2_mosi_roberta_cpf_cnUW_asf_s$s $s roberta-large --llrd 0.9 --warmup_steps 200 --epochs 20 \
      --max_len 64 --case_rule upper_words --asf 1 --asf_lambda 1.0
done

# --- encoder-family robustness (single seed, informational)
run P1_mosi_deberta_cpf_cnUW_s1111 1111 microsoft/deberta-v3-large --llrd 0.9 --warmup_steps 200 \
    --epochs 20 --max_len 64 --case_rule upper_words
run P2_mosi_xlnet_cpf_cnUW_s1111 1111 xlnet-base-cased --llrd 0.9 --warmup_steps 200 \
    --epochs 20 --max_len 64 --case_rule upper_words
echo "=== S2 DONE $(date -Is) ==="
