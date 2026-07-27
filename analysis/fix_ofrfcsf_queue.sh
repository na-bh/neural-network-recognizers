#!/bin/bash
# Sequential training queue for the three fix experiments: odds-first,
# repeat-01, compute-sqrt. For each task: 4 objectives x 4 architectures x
# 10 seeds (160 runs), then standard-baseline evaluation on the corrected
# test set, then full-sweep decomposition analysis. Each stage is
# resumable (single-objective scripts skip already-completed (arch,trial)
# pairs; standard-baseline and full-sweep scripts skip already-completed
# cells), so a crash/restart mid-run does not lose progress.
set -e
cd /mnt/linuxlab/home/reunbhandari/neural-network-recognizers
export PYTHONPATH=src:analysis

for task in odds_first repeat_01 compute_sqrt; do
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
echo "### QUEUE: ALL THREE TASKS COMPLETE -- $(date -u +%FT%TZ)"
echo "################################################################"
