#!/usr/bin/env bash
# GPU mutual exclusion: refuse to start a chain while another training process
# holds the accelerator.
#
# The 2026-11-24 S6 attempt died with "CUDA out of memory ... Process N has
# 10.16 GiB memory in use": two chains were running at the same time and the
# second one starved.  This guard makes that failure impossible by refusing to
# start instead of crashing halfway.
#
# Usage:  bash scripts/gpu_guard.sh <tag>   (exit 0 = free to start)
set -u

TAG=${1:-chain}
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
TRAINERS=$(pgrep -f "msa[.]train" | wc -l)
if [ "$TRAINERS" -gt 0 ]; then
  echo "[gpu_guard] 拒绝启动 $TAG：已有 $TRAINERS 个训练进程在跑"
  pgrep -af "msa[.]train" | head -3
  exit 1
fi
if [ "${USED:-0}" -gt 1500 ]; then
  echo "[gpu_guard] 拒绝启动 $TAG：GPU 已用 ${USED} MiB（可能残留）"
  exit 1
fi
echo "[gpu_guard] $TAG 可以启动（GPU 已用 ${USED} MiB）"
exit 0
