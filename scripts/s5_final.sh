#!/usr/bin/env bash
# S5 -- extend an arm to five seeds and produce the headline numbers.
#
# Usage: bash scripts/s5_final.sh <arm-spec> [<arm-spec> ...]
#
# An arm spec is   <tag-prefix>:<tokenizer>:<extra train args>
# for example
#   bash scripts/s5_final.sh B1_mosi_roberta_cpf_cnUW:roberta-large:--max_len 64 --case_rule upper_words
#
# The seeds 4444 and 5555 are added to the ones already run, so a cell is only
# trained when its result is missing; re-running a cell is always explicit
# (--force).
set -u
export PYTHONPATH=$HOME/MSA-SOTA/pylibs
export HF_HOME=$HOME/MSA-SOTA/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
cd $HOME/MSA-SOTA/code

RES=$HOME/MSA-SOTA/results
FORCE=0
[ "${1:-}" = "--force" ] && { FORCE=1; shift; }

run () {   # tag, seed, corpus, tokenizer, extra args...
  local tag=$1; shift
  local seed=$1; shift
  local corpus=$1; shift
  local tk=$1; shift
  local cell=$RES/$tag
  if [ -f "$cell/$(basename $tag)_result.csv" ] && [ "$FORCE" = "0" ]; then
    echo "--- 跳过已存在的 $tag"
    return 0
  fi
  rm -rf "$cell"; mkdir -p "$cell"
  echo "=== $tag (seed=$seed) $(date -Is) ==="
  /usr/bin/python3 -m msa.train --data $HOME/MSA-SOTA/data/$corpus --corpus $corpus \
      --tokenizer "$tk" --cache_dir $HOME/MSA-SOTA/hf_cache \
      --seed "$seed" --out "$cell" --tag "$tag" "$@" 2>&1 | \
      tee $HOME/MSA-SOTA/logs/$tag.log | \
      grep -E "^\[data\]|^\[best|^\[token|Traceback|Error"
}

for spec in "$@"; do
  prefix="${spec%%:*}"; rest="${spec#*:}"
  tk="${rest%%:*}"; extra="${rest#*:}"
  corpus="${prefix#*_}"; corpus="${corpus%%_*}"
  [ "$corpus" = "mosi" ] || [ "$corpus" = "mosei" ] || corpus=mosi
  for s in 1111 2222 3333 4444 5555; do
    run "${prefix}_s${s}" "$s" "$corpus" "$tk" $extra
  done
done
echo "=== S5 DONE $(date -Is) ==="
