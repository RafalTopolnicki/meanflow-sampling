#!/bin/bash
# Single-GPU eval-only launcher (no sudo, no TPU pod init).
set -e
export LOG_DIR=${LOG_DIR:-./logs}
export now=`date '+%Y%m%d_%H%M%S'`
export JOBNAME=${now}_${1:-eval}
export RUN_DIR=$LOG_DIR/$JOBNAME
mkdir -p ${RUN_DIR}

python3 main.py \
    --workdir=${RUN_DIR} \
    --config=configs/load_config.py:eval_b4 \
    2>&1 | tee -a $RUN_DIR/output.log
