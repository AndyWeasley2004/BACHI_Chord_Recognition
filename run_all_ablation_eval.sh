#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_all_ablation_eval.sh <data_name>

DATA_NAME=${1:?Specify data_name, e.g., classical or pop}

python eval.py checkpoints/${DATA_NAME}_baseline
python eval.py checkpoints/${DATA_NAME}_film_ctx
python eval.py checkpoints/${DATA_NAME}_film_kdec
python eval.py checkpoints/${DATA_NAME}_film_kdec_key
# python eval.py checkpoints/${DATA_NAME}_ht

echo "Completed evaluation for ${DATA_NAME}. See evaluation.txt in each checkpoint directory."