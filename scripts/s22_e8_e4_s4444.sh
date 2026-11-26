#!/usr/bin/env bash
# The two cells that were produced with the default maximum sequence length
# and without the case rule, and therefore cannot be compared with the rest
# of the arm, have to be produced again.
#
# The parameters are the ones recorded in the result files of the arms:
# roberta-large, batch 32, the maximum sequence length 64 and the case rule
# that folds shouted transcriptions.
set -u
export PYTHONPATH=$HOME/MSA-SOTA/code
export HF_HOME=$HOME/MSA-SOTA/hf_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
cd $HOME/MSA-SOTA

run () {
  local tag=$1 seed=$2 route=$3
  echo "=== START $tag $(date -Is)"
  python3 -m msa.train \
    --data $HOME/MSA-SOTA/data/mosi --corpus mosi \
    --tokenizer roberta-large --cache_dir $HOME/MSA-SOTA/hf_cache \
    --epochs 20 --lr_main 1e-3 --lr_bert 5e-5 --lr_schedule constant \
    --batch_size 32 --accum 1 --seed $seed --weight_decay 1e-4 \
    --warmup_steps 200 --llrd 0.9 --loss l1 --max_len 64 \
    --case_rule upper_words \
    --binary_head 1 --lambda_binary 0.1 --context 1 --context_route $route \
    --out $HOME/MSA-SOTA/results/$tag --tag $tag \
    > $HOME/MSA-SOTA/logs/$tag.log 2>&1
  local rc=$?
  grep -E "^\[best|Traceback|Error" $HOME/MSA-SOTA/logs/$tag.log
  echo "[EXIT_STATUS] $tag rc=$rc $(date -Is)"
  while pgrep -f "msa[.]train" > /dev/null; do sleep 15; done
  sleep 10
}

run E8_mosi_bhead_ctxreg_s4444 4444 reg
run E4_mosi_bhead_ctx_s4444 4444 both
echo "=== S22 DONE $(date -Is) ==="
