#!/bin/bash
# Sequential training queue for binary-multiplication then stack-manipulation,
# chained to run AFTER the currently-running odds-first/repeat-01/compute-sqrt
# queue (fix_ofrfcsf_queue.sh, PID passed as $1) fully exits. Each stage is
# resumable (single-objective scripts skip already-completed (arch,trial)
# pairs; standard-baseline and full-sweep scripts skip already-completed
# cells).
set -e
PRIOR_QUEUE_PID="$1"
cd /mnt/linuxlab/home/reunbhandari/neural-network-recognizers
export PYTHONPATH=src:analysis

if [ -n "$PRIOR_QUEUE_PID" ]; then
  echo "### waiting for prior queue (PID ${PRIOR_QUEUE_PID}) to finish -- $(date -u +%FT%TZ)"
  while kill -0 "$PRIOR_QUEUE_PID" 2>/dev/null; do
    sleep 30
  done
  echo "### prior queue (PID ${PRIOR_QUEUE_PID}) has exited -- $(date -u +%FT%TZ)"
fi

for task in binary_multiplication stack_manipulation; do
  for obj in rec rec+lm rec+ns rec+lm+ns; do
    echo ""
    echo "################################################################"
    echo "### QUEUE: task=${task} objective=${obj} -- $(date -u +%FT%TZ)"
    echo "################################################################"
    python analysis/fix_train_models_${task}_single_objective.py --objective "$obj"
  done
  echo ""
  echo "################################################################"
  echo "### QUEUE: task=${task} standard-baseline eval -- $(date -u +%FT%TZ)"
  echo "################################################################"
  python analysis/fix_standard_baseline_on_corrected_test_all_objectives_${task}.py

  echo ""
  echo "################################################################"
  echo "### QUEUE: task=${task} full-sweep analysis -- $(date -u +%FT%TZ)"
  echo "################################################################"
  python analysis/fix_full_sweep_analysis_${task}.py
done

echo ""
echo "################################################################"
echo "### QUEUE: BINARY-MULTIPLICATION + STACK-MANIPULATION COMPLETE -- $(date -u +%FT%TZ)"
echo "################################################################"
