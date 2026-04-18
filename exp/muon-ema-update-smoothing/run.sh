#!/bin/bash
set -e

# Activate environment
source /inspire/hdd/global_user/lipengfei-253107010003/miniconda3/etc/profile.d/conda.sh
conda activate marti

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

echo "=== Running muon-ema-update-smoothing experiments ==="
python exp/run_experiments.py exp/muon-ema-update-smoothing/muon-ema-update-smoothing.json
echo "=== Experiments complete ==="

# Sync offline wandb runs
echo "=== Syncing wandb runs ==="
for run_dir in "$PROJECT_DIR"/wandb/offline-run-*; do
    if [ -d "$run_dir" ]; then
        echo "Syncing $run_dir ..."
        wandb sync "$run_dir" --timeout 120 || echo "Warning: failed to sync $run_dir"
    fi
done
echo "=== Done ==="
