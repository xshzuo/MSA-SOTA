#!/usr/bin/env bash
# Poll the remote chain with short SSH connections (a long-lived SSH session
# gets dropped and loses the waiter's output).
set -u
DUR=${1:-240}
INTERVAL=${2:-300}
CHAIN=${3:-s8_chain}          # the chain log to watch (without .out)
END=$(( $(date +%s) + DUR ))
while [ "$(date +%s)" -lt "$END" ]; do
  # the chain name has to be substituted locally: the remote command runs in
  # single quotes, so a remote-side $CHAIN would expand to nothing there
  OUT=$(sshpass -p "${MSA_SSH_PASSWORD:-}" ssh -n -o StrictHostKeyChecking=no -o ConnectTimeout=20 \
        "${MSA_SSH_USER:-alan}@${MSA_SSH_HOST:-localhost}" \
        "cd ~/MSA-SOTA && \
         echo \"完成格 \$(grep -c '^\[EXIT_STATUS\]' logs/$CHAIN.out 2>/dev/null)\"; \
         echo \"失败 \$(grep -c '^!!!' logs/$CHAIN.out 2>/dev/null)\"; \
         ps -eo cmd | grep '[m]sa[.]train' | grep -o '\-\-tag [A-Za-z0-9_]*' | head -1; \
         grep -E '^\[best' logs/$CHAIN.out 2>/dev/null | tail -3; \
         date +%H:%M" 2>&1)
  echo "$OUT"
  if echo "$OUT" | grep -qE "=== S[0-9] DONE"; then echo "链已完成"; exit 0; fi
  sleep "$INTERVAL"
done
echo "轮询窗口结束"
