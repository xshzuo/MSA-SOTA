#!/usr/bin/env bash
# The follow-up sweep, queued behind the S6 chain.
#
# What the S6 results established (MOSI, 3 paired seeds):
#   * the context routing passed its pre-registered line L4:
#     E8 (routed) - E4 (context in both heads) = +1.46 pp Acc-2 (t=+3.27,
#     positive on all three seeds) and a better MAE;
#   * d_model=128 bought nothing (E9-E8 = -0.15, E5-E1 = +0.10) -> dropped;
#   * the decision head alone improves the intensity estimate
#     (MAE -0.017, t=-4.04) but not the binary accuracy.
#
# So the remaining gap is the encoder capacity, and the two arms that combine
# the confirmed levers are queued here:
#   E10  DeBERTa-v3-large + the decision head + the routed context
#   E11  E8 with 30 epochs (the best epoch was often 12-18, so the schedule
#        may have been the binding constraint)
#   C5   MOSEI with the decision head and the routed context
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
SEEDS="1111 2222 3333"

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
  /usr/bin/python3 -m msa.train --data $HOME/MSA-SOTA/data/$corpus --corpus $corpus \
      --tokenizer "$tok" --cache_dir $HOME/MSA-SOTA/hf_cache \
      --seed "$seed" --out "$cell" --tag "$tag" \
      --llrd 0.9 --warmup_steps 200 --max_len 64 --case_rule upper_words "$@" \
      > "$LOGS/$tag.log" 2>&1
  local rc=$?
  grep -E "^\[best|^\[token|Traceback|Error" "$LOGS/$tag.log"
  echo "[EXIT_STATUS] $tag rc=$rc $(date -Is)"
  if [ "$rc" != "0" ]; then echo "!!! $tag 失败 rc=$rc，该格作废"; fi
}

for s in $SEEDS; do
  run E10_mosi_deberta_bhead_ctxreg_s$s $s microsoft/deberta-v3-large mosi --epochs 20 \
      --binary_head 1 --lambda_binary 0.1 --context 1 --context_route reg
done
for s in $SEEDS; do
  run E11_mosi_bhead_ctxreg_ep30_s$s $s roberta-large mosi --epochs 30 \
      --binary_head 1 --lambda_binary 0.1 --context 1 --context_route reg
done
for s in $SEEDS; do
  run C5_mosei_bhead_ctxreg_s$s $s roberta-large mosei --epochs 20 \
      --binary_head 1 --lambda_binary 0.1 --context 1 --context_route reg
done

echo "=== S7 DONE $(date -Is) ==="
