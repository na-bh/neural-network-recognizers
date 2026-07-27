#!/bin/bash
# CPF3: sequential training queue for the marked-reversal contrastive-pairing
# experiment. 4 objectives x 4 architectures x 10 seeds = 160 runs, each
# using --pair-id-file so batches respect pair boundaries. Resumable
# (single-objective script skips already-completed (arch,trial) pairs).
set -e
cd /mnt/linuxlab/home/reunbhandari/neural-network-recognizers
export PYTHONPATH=src:analysis

for obj in rec rec+lm rec+ns rec+lm+ns; do
  echo ""
  echo "################################################################"
  echo "### CPF3 QUEUE: objective=${obj} -- $(date -u +%FT%TZ)"
  echo "################################################################"
  python analysis/fix_train_models_marked_reversal_contrastive_single_objective.py --objective "$obj"
done

echo ""
echo "################################################################"
echo "### CPF3 QUEUE: ALL 4 OBJECTIVES COMPLETE -- $(date -u +%FT%TZ)"
echo "################################################################"
