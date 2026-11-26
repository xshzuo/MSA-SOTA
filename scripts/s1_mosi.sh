#!/usr/bin/env bash
# S1 -- the minimal clean-implementation model on CMU-MOSI.
#
#   A0  BERT-base, flat learning rate, no warmup   (the equivalence baseline)
#   A1  A0 + CPF (layer-wise LR decay + warmup)   (decision line L2)
#
# Gate (docs/03_EXECUTION_PLAN.md, stage S1): Acc-2(all) >= 83 and MAE <= 0.75.
set -u
export PYTHONPATH=$HOME/MSA-SOTA/pylibs
export HF_HOME=$HOME/MSA-SOTA/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
cd $HOME/MSA-SOTA/code

DATA=$HOME/MSA-SOTA/data/mosi
RES=$HOME/MSA-SOTA/results
mkdir -p $RES

run () {   # tag, seed, extra args...
  local tag=$1; shift
  local seed=$1; shift
  local cell=$RES/$tag
  rm -rf "$cell"; mkdir -p "$cell"
  echo "=== $tag (seed=$seed) $(date -Is) ==="
  /usr/bin/python3 -m msa.train --data $DATA --corpus mosi \
      --tokenizer bert-base-uncased --cache_dir $HOME/MSA-SOTA/hf_cache \
      --seed $seed --out "$cell" --tag $tag "$@" 2>&1 | tee $HOME/MSA-SOTA/logs/$tag.log | \
      grep -E "^\[model\]|^\[optim\]|^\[param group\]|^\[epoch|^\[best|^\[token|^\[done\]|Traceback|Error"
}

# A0: the flat schedule -- llrd 1.0 and no warmup reproduce "one learning rate
# for the whole encoder", which is what the released recipes do.
run A0_mosi_bert_flat_s1111 1111 --llrd 1.0 --warmup_steps 0 --epochs 20
# A1: CPF
run A1_mosi_bert_cpf_s1111 1111 --llrd 0.9 --warmup_steps 200 --epochs 20
echo "=== S1 DONE $(date -Is) ==="
