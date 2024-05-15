#!/bin/bash
# Usage: CUDA_VISIBLE_DEVICES="..." checkpoint_eval.sh <script> <checkpoints dir>
if [ "$#" -ne 2 ]; then
    echo "Usage: CUDA_VISIBLE_DEVICES=\"...\" checkpoint_eval.sh <script> <checkpoints dir>"
    exit 1
fi

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "Please set CUDA_VISIBLE_DEVICES"
    exit 1
fi

SCRIPT=$1
CHECKPOINT_DIR=$2
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr "," "\n" | wc -l)

# ENV vars with default values
TEMPERATURE=${TEMPERATURE:-0.0}
COMPLETION_LIMIT=${COMPLETION_LIMIT:-1}

for CHECKPOINT in $(ls $CHECKPOINT_DIR); do
    echo "Evaluating $CHECKPOINT"
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python $SCRIPT \
      --model $CHECKPOINT_DIR/$CHECKPOINT \
      --temperature $TEMPERATURE \
      --completion-limit $COMPLETION_LIMIT \
      --num-gpus $NUM_GPUS \
      --output-dir "${CHECKPOINT_DIR}/${CHECKPOINT}_eval_temp${TEMPERATURE}_comps${COMPLETION_LIMIT}"
done
