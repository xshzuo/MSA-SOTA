#!/usr/bin/env bash
# The endgame sweep.
#
# What the results so far say (MOSI, 3 seeds, paired):
#   E6  deberta + the decision head            Acc-2 87.32  Acc-7 49.42  MAE 0.5830
#   E10 E6 + the context routed to the
#       regression head only                   Acc-2 86.1x  Acc-7 ~52    MAE ~0.57
# The two arms are strong on different axes.  The routed context does not feed
# the decision head, so the Acc-2 difference is not the routing itself: the
# context stream's gradient disturbs the *shared* text encoder.  The fix that
# follows from that reading is to strengthen the decision head's weight in the
# joint loss, so that the binary objective keeps the trunk in line.
#
#   E13  E10 with lambda_binary 0.5
#   E14  E6  with lambda_binary 0.5
#   E15  the winner of E13/E14 extended to five seeds
#   C4   MOSEI: deberta + the decision head (one seed first)
#   C5   MOSEI: the confirmed recipe with the routed context
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

bash $HOME/MSA-SOTA/scripts/gpu_guard.sh s10 || exit 1

# the decision head's weight in the joint loss
for s in 1111 2222 3333; do
  run E13_mosi_deberta_bhead_ctxreg_lb05_s$s $s microsoft/deberta-v3-large mosi --epochs 20 \
      --binary_head 1 --lambda_binary 0.5 --context 1 --context_route reg
done
for s in 1111 2222 3333; do
  run E14_mosi_deberta_bhead_lb05_s$s $s microsoft/deberta-v3-large mosi --epochs 20 \
      --binary_head 1 --lambda_binary 0.5
done

# MOSEI
for s in 1111; do
  run C4_mosei_deberta_bhead_s$s $s microsoft/deberta-v3-large mosei --epochs 20 \
      --binary_head 1 --lambda_binary 0.1
done
for s in 1111 2222 3333; do
  run C5_mosei_bhead_ctxreg_s$s $s roberta-large mosei --epochs 20 \
      --binary_head 1 --lambda_binary 0.1 --context 1 --context_route reg
done

echo "=== S10 DONE $(date -Is) ==="
