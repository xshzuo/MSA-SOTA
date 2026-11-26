#!/usr/bin/env bash
# The last queue.  Two things are left to try.
#
# 1) The prior-neutral objective.  MOSI's training split has a positive rate
#    of 0.5701 (the negative side carries 0.4299), so an unweighted L1 loss
#    minimises the *plain* median of the training labels and places the
#    regression output high.  The L1 objective is minimised by the weighted
#    median, so weighting the negative side by n_pos/n_neg = 1.3261 moves the
#    output along the label axis.  This uses no test label: it is the usual
#    class-balanced objective, and it is the only lever that can move the
#    Acc-2 threshold without changing the metric definition.  MOSEI's
#    training positive rate is 0.7098, so its balance weight would be 2.4458
#    and it is not used here.
#
# 2) The MOSEI arm with the stronger encoder, the only remaining candidate
#    that could move the MOSEI Acc-2 columns.
#
# The earlier attempt at this queue failed twice for the same reason: a chain
# that the GPU guard refuses exits immediately, and a supervisor that then
# starts the next chain collides with a cell that is still doing its final
# evaluation.  So every launch goes through a retry loop, and the next cell
# only starts once the previous one has left no process behind.
set -u
cd ~/MSA-SOTA
export PYTHONPATH=$HOME/MSA-SOTA/pylibs
export HF_HOME=$HOME/MSA-SOTA/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0

BAL=1.3261

run_cell () {   # tag seed tokenizer corpus extra...
  local tag=$1; shift
  local seed=$1; shift
  local tok=$1; shift
  local corpus=$1; shift
  local cell=$HOME/MSA-SOTA/results/$tag
  if ls "$cell" 2>/dev/null | grep -q "_result.csv"; then
    echo "--- 跳过已存在的 $tag"; return 0
  fi
  for attempt in $(seq 1 20); do
    if bash $HOME/MSA-SOTA/scripts/gpu_guard.sh "$tag"; then break; fi
    echo "[retry] $tag 闸门拒绝，等待后重试 (第 $attempt 次)"
    sleep 120
  done
  rm -rf "$cell"; mkdir -p "$cell"
  echo "=== $tag (seed=$seed, $tok, $corpus) $(date -Is) ==="
  local extra=""
  if [ "$tok" = "microsoft/deberta-v3-large" ]; then extra="--batch_size 8 --accum 4"; fi
  cd $HOME/MSA-SOTA/code
  /usr/bin/python3 -m msa.train --data $HOME/MSA-SOTA/data/$corpus --corpus $corpus \
      --tokenizer "$tok" --cache_dir $HOME/MSA-SOTA/hf_cache \
      --seed "$seed" --out "$cell" --tag "$tag" \
      --llrd 0.9 --warmup_steps 200 --max_len 64 --case_rule upper_words $extra "$@" \
      > $HOME/MSA-SOTA/logs/$tag.log 2>&1
  local rc=$?
  grep -E "^\[best|Traceback|Error" $HOME/MSA-SOTA/logs/$tag.log
  echo "[EXIT_STATUS] $tag rc=$rc $(date -Is)"
  while pgrep -f "msa[.]train" > /dev/null; do sleep 15; done
  sleep 10
}

# 0) smoke test of the weighted loss: one epoch, and the queue stops if it
#    does not come out clean
echo "=== 冒烟测试：加权损失 1 epoch $(date -Is) ==="
cd $HOME/MSA-SOTA/code
/usr/bin/python3 -m msa.train --data $HOME/MSA-SOTA/data/mosi --corpus mosi \
  --tokenizer microsoft/deberta-v3-large --cache_dir $HOME/MSA-SOTA/hf_cache \
  --seed 1111 --out /tmp/smoke_bal --tag SMOKE_bal --llrd 0.9 --warmup_steps 200 \
  --max_len 64 --case_rule upper_words --batch_size 8 --accum 4 --epochs 1 \
  --binary_head 1 --lambda_binary 0.1 --neg_weight $BAL \
  > $HOME/MSA-SOTA/logs/SMOKE_bal.log 2>&1
src=$?
echo "[EXIT_STATUS] SMOKE_bal rc=$src"
grep -E "^\[epoch|^\[best|Traceback|Error" $HOME/MSA-SOTA/logs/SMOKE_bal.log | tail -5
if [ "$src" != "0" ]; then
  echo "!!! 冒烟测试失败，队列中止"
  exit 1
fi
rm -rf /tmp/smoke_bal

# 1) the balanced objective on the Acc-2 arm (E6's recipe)
for s in 1111 2222 3333; do
  run_cell E19_mosi_deberta_bhead_bal_s$s $s microsoft/deberta-v3-large mosi \
      --epochs 20 --binary_head 1 --lambda_binary 0.1 --neg_weight $BAL
done
# 2) the balanced objective on the Acc-7 arm (E17's recipe)
for s in 1111 2222; do
  run_cell E20_mosi_deberta_lb05_mse_bal_s$s $s microsoft/deberta-v3-large mosi \
      --epochs 20 --binary_head 1 --lambda_binary 0.5 --loss mse --neg_weight $BAL
done
# 3) MOSEI with the stronger encoder
run_cell C4_mosei_deberta_bhead_s1111 1111 microsoft/deberta-v3-large mosei \
    --epochs 20 --binary_head 1 --lambda_binary 0.1

echo "=== S13 DONE $(date -Is) ==="
