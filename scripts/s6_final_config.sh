#!/usr/bin/env bash
# S6 -- the final-configuration candidates, plus the MOSEI encoder cell that
# died silently on 2026-11-24.
#
# Serial by construction.  Three failure modes are closed here:
#   * two chains running at once starved the second one (CUDA OOM) -> the GPU
#     guard refuses to start while another training process is alive;
#   * a cell dying silently looked like a result -> every cell now records its
#     exit code, and a non-zero code marks the cell invalid;
#   * deberta-v3-large ships fp16 weights and failed with "expected scalar
#     type Half but found Float" -> the encoder is now loaded as float32.
#
#   E1  the confirmed best arm + the joint decision head
#   E4  E1 + the context stream
#   E5  E1 + a wider shared space (d_model 128)
#   E6  E1 with deberta-v3-large as the text encoder
#   E7  E6 + the context stream
#   E8  E1 + the context stream, routed to the regression head only
#   E9  E8 + the wider shared space
#   C3  MOSEI with deberta-v3-large
#   C4  MOSEI with deberta-v3-large + the joint decision head
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

bash $HOME/MSA-SOTA/scripts/gpu_guard.sh s6 || exit 1

# --- MOSI, the final candidates -------------------------------------------
for s in $SEEDS; do
  run E1_mosi_bhead_s$s $s roberta-large mosi --epochs 20 --binary_head 1 --lambda_binary 0.1
done
for s in $SEEDS; do
  run E4_mosi_bhead_ctx_s$s $s roberta-large mosi --epochs 20 --binary_head 1 --lambda_binary 0.1 \
      --context 1
done
for s in $SEEDS; do
  run E8_mosi_bhead_ctxreg_s$s $s roberta-large mosi --epochs 20 --binary_head 1 --lambda_binary 0.1 \
      --context 1 --context_route reg
done
for s in $SEEDS; do
  run E5_mosi_bhead_d128_s$s $s roberta-large mosi --epochs 20 --binary_head 1 --lambda_binary 0.1 \
      --d_model 128
done
for s in $SEEDS; do
  run E9_mosi_bhead_ctxreg_d128_s$s $s roberta-large mosi --epochs 20 --binary_head 1 --lambda_binary 0.1 \
      --context 1 --context_route reg --d_model 128
done
for s in $SEEDS; do
  run E6_mosi_deberta_bhead_s$s $s microsoft/deberta-v3-large mosi --epochs 20 --binary_head 1 \
      --lambda_binary 0.1
done
for s in $SEEDS; do
  run E7_mosi_deberta_bhead_ctx_s$s $s microsoft/deberta-v3-large mosi --epochs 20 --binary_head 1 \
      --lambda_binary 0.1 --context 1
done

# --- MOSEI: the encoder cell that died silently, and the decision head -----
for s in $SEEDS; do
  run C3_mosei_deberta_cpf_cnUW_s$s $s microsoft/deberta-v3-large mosei --epochs 20
done
for s in $SEEDS; do
  run C4_mosei_deberta_bhead_s$s $s microsoft/deberta-v3-large mosei --epochs 20 \
      --binary_head 1 --lambda_binary 0.1
done

echo "=== S6 DONE $(date -Is) ==="
