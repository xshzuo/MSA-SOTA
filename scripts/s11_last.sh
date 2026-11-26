#!/usr/bin/env bash
# The last sweep.  What the previous rounds established:
#
#   * Acc-7 and Acc-2 trade off through the shared encoder's output scale.
#     Three independent interventions move them in opposite directions:
#     the loss type (L1 <-> MSE), the decision head's weight in the joint loss
#     (lambda_binary 0.1 <-> 0.5) and the measured prediction scale ratio.
#     The published combination (Acc-2 >= 87.51 AND Acc-7 >= 50.34) is not on
#     our frontier, so the remaining runs look for the best point *on* it.
#
#   E15  the Acc-2 arm (no context) with an MSE loss: the MSE loss was the
#        only intervention that raised Acc-2 by more than a point, and it did
#        so while the Acc-7 fell -- so it is worth testing without the
#        context, where the Acc-7 has more headroom.
#   E16  E15 plus the routed context: the context is what makes the Acc-7
#        reachable, and the routing is what keeps it from wrecking the
#        polarity.
#   E6x  the two extra seeds of E6, the current best Acc-2 arm (the final
#        numbers need five seeds).
#   C5   MOSEI with the decision head and the routed context.
#   C4   MOSEI with deberta-v3-large and the decision head.
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

bash $HOME/MSA-SOTA/scripts/gpu_guard.sh s11 || exit 1

# the Acc-2 arm: no context, MSE loss
run E15_mosi_deberta_mse_s1111 1111 microsoft/deberta-v3-large mosi --epochs 20 \
    --binary_head 1 --lambda_binary 0.1 --loss mse
# ... plus the routed context
run E16_mosi_deberta_mse_ctxreg_s1111 1111 microsoft/deberta-v3-large mosi --epochs 20 \
    --binary_head 1 --lambda_binary 0.1 --context 1 --context_route reg --loss mse
# the two extra seeds of the confirmed best Acc-2 arm
run E6_mosi_deberta_bhead_s4444 4444 microsoft/deberta-v3-large mosi --epochs 20 \
    --binary_head 1 --lambda_binary 0.1
run E6_mosi_deberta_bhead_s5555 5555 microsoft/deberta-v3-large mosi --epochs 20 \
    --binary_head 1 --lambda_binary 0.1
# MOSEI
for s in 1111 2222 3333; do
  run C5_mosei_bhead_ctxreg_s$s $s roberta-large mosei --epochs 20 \
      --binary_head 1 --lambda_binary 0.1 --context 1 --context_route reg
done
run C4_mosei_deberta_bhead_s1111 1111 microsoft/deberta-v3-large mosei --epochs 20 \
    --binary_head 1 --lambda_binary 0.1

echo "=== S11 DONE $(date -Is) ==="
