#!/usr/bin/env bash
# Re-run the cells that failed because the large encoder did not fit:
#   * deberta-v3-large in fp32 needs > 15.2 GiB at batch 32 on a 15.47 GiB card
#   * the fix is gradient accumulation: micro-batch 8 x accum 4 = the same
#     effective batch (32) and the same number of optimizer steps, so the arm
#     stays comparable with the roberta arms
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
      --llrd 0.9 --warmup_steps 200 --max_len 64 --case_rule upper_words \
      --batch_size 8 --accum 4 "$@" \
      > "$LOGS/$tag.log" 2>&1
  local rc=$?
  grep -E "^\[best|^\[token|Traceback|Error" "$LOGS/$tag.log"
  echo "[EXIT_STATUS] $tag rc=$rc $(date -Is)"
  if [ "$rc" != "0" ]; then echo "!!! $tag 失败 rc=$rc，该格作废"; fi
}

run_plain () {   # like run() but without the gradient accumulation
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

bash $HOME/MSA-SOTA/scripts/gpu_guard.sh s8 || exit 1

for s in $SEEDS; do
  run E6_mosi_deberta_bhead_s$s $s microsoft/deberta-v3-large mosi --epochs 20 \
      --binary_head 1 --lambda_binary 0.1
done
for s in $SEEDS; do
  run E7_mosi_deberta_bhead_ctx_s$s $s microsoft/deberta-v3-large mosi --epochs 20 \
      --binary_head 1 --lambda_binary 0.1 --context 1
done
for s in $SEEDS; do
  run E10_mosi_deberta_bhead_ctxreg_s$s $s microsoft/deberta-v3-large mosi --epochs 20 \
      --binary_head 1 --lambda_binary 0.1 --context 1 --context_route reg
done
for s in $SEEDS; do
  run C3_mosei_deberta_cpf_cnUW_s$s $s microsoft/deberta-v3-large mosei --epochs 20
done
for s in $SEEDS; do
  run C4_mosei_deberta_bhead_s$s $s microsoft/deberta-v3-large mosei --epochs 20 \
      --binary_head 1 --lambda_binary 0.1
done

# the two arms that the earlier S7 attempt did not finish (it was killed
# mid-cell and left a child holding the accelerator)
for s in $SEEDS; do
  run_plain E11_mosi_bhead_ctxreg_ep30_s$s $s roberta-large mosi --epochs 30 \
      --binary_head 1 --lambda_binary 0.1 --context 1 --context_route reg
done
for s in $SEEDS; do
  run_plain C5_mosei_bhead_ctxreg_s$s $s roberta-large mosei --epochs 20 \
      --binary_head 1 --lambda_binary 0.1 --context 1 --context_route reg
done

echo "=== S8 DONE $(date -Is) ==="
