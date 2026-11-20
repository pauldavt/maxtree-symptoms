#!/usr/bin/env bash
# Build mta.so and train all models for one cultivar.
#
# Usage: ./run_pipeline.sh <cultivar>
#   e.g. ./run_pipeline.sh Rudolph
#
# Set DATA_DIR to use another data folder than the default in config.py.
# mta.so is built with build_mta.sh; models are saved in runs/<cultivar>_<time>/
# in the current working directory, without the time in their file names.
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <cultivar>" >&2
    exit 1
fi
cultivar=$1

here=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)

"$here/build_mta.sh"

# A new folder per run: mkdir (without -p) fails if the folder exists, so runs
# started in the same second get _2, _3, ... instead of sharing a folder.
base="runs/${cultivar}_$(date -u +%Y%m%d_%H%M%S)"
mkdir -p runs
run_dir=$base
i=1
until mkdir "$run_dir" 2>/dev/null; do
    if [ ! -e "$run_dir" ]; then
        mkdir "$run_dir"  # failed for another reason: show the error and stop
    fi
    i=$((i + 1))
    run_dir="${base}_$i"
done
common=(--cultivar "$cultivar" --output-dir "$run_dir" --no-timestamp)
if [ -n "${DATA_DIR:-}" ]; then
    common+=(--data-dir "$DATA_DIR")
fi

python "$here/finetune.py" "${common[@]}"
checkpoint="$run_dir/model_${cultivar}.pt"

python "$here/train_logreg.py" patch "${common[@]}" --checkpoint "$checkpoint"
patch_logreg="$run_dir/patch_logreg_${cultivar}.npz"

python "$here/train_logreg.py" component "${common[@]}" --patch-logreg "$patch_logreg"

echo "Models saved in $run_dir"
