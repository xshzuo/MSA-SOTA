#!/usr/bin/env bash
# The cells that still have to be produced, with a check that the arguments
# match the ones recorded in the result files of the arms.
#
# The earlier version of this chain left --max_len and --case_rule at their
# defaults, which produced cells that cannot be compared with the ones
# already on disk.  The recorded values are max_len 64 and the case rule
# that folds shouted transcriptions, and the cells produced with the
# defaults have to be thrown away.
set -u
export PYTHONPATH=$HOME/MSA-SOTA/code
export HF_HOME=$HOME/MSA-SOTA/hf_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
cd $HOME/MSA-SOTA

COMMON="--epochs 20 --lr_main 1e-3 --lr_bert 5e-5 --lr_schedule constant
        --weight_decay 1e-4 --warmup_steps 200 --llrd 0.9 --max_len 64
        --case_rule upper_words --context 1"

run () {
  local tag=$1 seed=$2 tok=$3 bs=$4 acc=$5 loss=$6 lam=$7 route=$8
  echo "=== START $tag $(date -Is)"
  python3 -m msa.train \
    --data $HOME/MSA-SOTA/data/mosi --corpus mosi \
    --tokenizer "$tok" --cache_dir $HOME/MSA-SOTA/hf_cache \
    $COMMON --batch_size $bs --accum $acc --seed $seed --loss $loss \
    --binary_head 1 --lambda_binary $lam --context_route $route \
    --out $HOME/MSA-SOTA/results/$tag --tag $tag \
    > $HOME/MSA-SOTA/logs/$tag.log 2>&1
  local rc=$?
  grep -E "^\[best|Traceback|Error" $HOME/MSA-SOTA/logs/$tag.log
  echo "[EXIT_STATUS] $tag rc=$rc $(date -Is)"
  while pgrep -f "msa[.]train" > /dev/null; do sleep 15; done
  sleep 10
}

run E8_mosi_bhead_ctxreg_s4444 4444 roberta-large 32 1 l1 0.1 reg
run E8_mosi_bhead_ctxreg_s5555 5555 roberta-large 32 1 l1 0.1 reg
run E4_mosi_bhead_ctx_s4444 4444 roberta-large 32 1 l1 0.1 both
run E4_mosi_bhead_ctx_s5555 5555 roberta-large 32 1 l1 0.1 both
run E18_mosi_deberta_lb05_mse_ctxreg_s1111 1111 microsoft/deberta-v3-large 8 4 mse 0.5 reg
echo "=== S22 DONE $(date -Is) ==="
