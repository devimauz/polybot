#!/usr/bin/env bash
# Watch twap_bot.log for real events (reliable on Windows: log freshness, not kill -0).
# Durable: prints NEW matching lines as they appear and keeps running; exits only
# when the log goes stale (>150s with no new lines = possible death).
LOG=twap_bot.log
NEXT_LINE=$(wc -l < "$LOG")
PATTERN='PAPER BUY|EXIT\(|SETTLE|ERROR|CRITICAL|Traceback|session_pnl=[1-9]|session_pnl=-[1-9]'
LAST_STALE=0
while :; do
  TOTAL=$(wc -l < "$LOG")
  if [ "$TOTAL" -gt "$NEXT_LINE" ]; then
    HITS=$(sed -n "$((NEXT_LINE+1)),${TOTAL}p" "$LOG" | grep -aE "$PATTERN")
    NEXT_LINE=$TOTAL
    if [ -n "$HITS" ]; then
      echo "WATCH: event at $(date +%H:%M:%S)"
      echo "$HITS"
    fi
    LAST_STALE=0
  fi
  # log stuck for >150s AND no new lines -> possible death
  NOW=$(date +%s)
  MTIME=$(stat -c %Y "$LOG" 2>/dev/null || stat -f %m "$LOG" 2>/dev/null)
  if [ -n "$MTIME" ] && [ $((NOW-MTIME)) -gt 150 ]; then
    LINES=$(wc -l < "$LOG")
    if [ "$LINES" -eq "$LAST_STALE" ]; then
      STALE=$((NOW-MTIME))
      echo "WATCH: log stale ${STALE}s, no new lines (possible death)"
      tail -n 30 "$LOG"
      exit 0
    fi
    LAST_STALE=$LINES
  else
    LAST_STALE=0
  fi
  sleep 15
done
