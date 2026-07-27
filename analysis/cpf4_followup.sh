#!/bin/bash
# Waits for CPF3's queue process to finish, then runs CPF4 (full-sweep
# decomposition analysis) automatically.
PRIOR_PID="$1"
cd /mnt/linuxlab/home/reunbhandari/neural-network-recognizers
export PYTHONPATH=src:analysis

if [ -n "$PRIOR_PID" ]; then
  echo "### waiting for CPF3 queue (PID ${PRIOR_PID}) to finish -- $(date -u +%FT%TZ)"
  while kill -0 "$PRIOR_PID" 2>/dev/null; do
    sleep 30
  done
  echo "### CPF3 queue (PID ${PRIOR_PID}) has exited -- $(date -u +%FT%TZ)"
fi

echo ""
echo "################################################################"
echo "### CPF4: full-sweep decomposition analysis -- $(date -u +%FT%TZ)"
echo "################################################################"
python analysis/fix_full_sweep_analysis_marked_reversal_contrastive.py

echo ""
echo "################################################################"
echo "### CPF4 COMPLETE -- $(date -u +%FT%TZ)"
echo "################################################################"
