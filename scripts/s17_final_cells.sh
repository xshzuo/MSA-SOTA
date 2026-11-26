#!/usr/bin/env bash
# The three cells that still have to be produced, one after the other.
#
#   E8  seed 4444 and 5555  the headline routing arm, extended to five seeds
#                            so that the paired test has more power than the
#                            three seeds it had
#   E18 seed 1111           the first run of that cell was removed from disk
#                            by a later retry of the same cell, so it has to
#                            be produced again for the three-seed mean
#
# The parameters are the ones recorded in the result file of the arm, not a
# hand-written guess: E8 uses roberta-large with batch 32, E18 uses
# deberta-v3-large with batch 8 and gradient accumulation 4.
set -u
export PYTHONPATH=$HOME/MSA-SOTA/code
export HF_HOME=$HOME/MSA-SOTA/hf_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
cd $HOME/MSA-SOTA

run () {
  local tag=$1 seed=$2 tok=$3 bs=$4 acc=$5 loss=$6 lam=$7
  echo "=== START $tag $(date -Is)"
  python3 -m msa.train \
    --data $HOME/MSA-SOTA/data/mosi --corpus mosi \
    --tokenizer "$tok" --cache_dir $HOME/MSA-SOTA/hf_cache \
    --epochs 20 --lr_main 1e-3 --lr_bert 5e-5 --lr_schedule constant \
    --batch_size $bs --accum $acc --seed $seed --weight_decay 1e-4 \
    --warmup_steps 200 --llrd 0.9 --loss $loss \
    --binary_head 1 --lambda_binary $lam --context 1 --context_route reg \
    --out $HOME/MSA-SOTA/results/$tag --tag $tag \
    > $HOME/MSA-SOTA/logs/$tag.log 2>&1
  local rc=$?
  grep -E "^\[best|Traceback|Error" $HOME/MSA-SOTA/logs/$tag.log
  echo "[EXIT_STATUS] $tag rc=$rc $(date -Is)"
  while pgrep -f "msa[.]train" > /dev/null; do sleep 15; done
  sleep 10
}

run E8_mosi_bhead_ctxreg_s4444 4444 roberta-large 32 1 l1 0.1
run E8_mosi_bhead_ctxreg_s5555 5555 roberta-large 32 1 l1 0.1
run E18_mosi_deberta_lb05_mse_ctxreg_s1111 1111 microsoft/deberta-v3-large 8 4 mse 0.5
echo "=== S17 DONE $(date -Is) ==="
