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
BATCH_SIZE=${BATCH_SIZE:-256}
EXEC_BATCH_SIZE=${EXEC_BATCH_SIZE:-$(nproc)}

# string version of TEMPERATURE, without the dot
TEMPERATURE_STR=$(echo $TEMPERATURE | tr -d . | tr - _)

SCRIPT_NAME=$(basename $SCRIPT)
SCRIPT_NAME=${SCRIPT_NAME%.*}

for CHECKPOINT in $(ls $CHECKPOINT_DIR); do
    echo "Evaluating $CHECKPOINT"
    # get checkpoint number, after "-"
    CHECKPOINT_NUM=${CHECKPOINT#*-}
    OUTPUT_PATH="${CHECKPOINT_DIR}/${CHECKPOINT}/${SCRIPT_NAME}_${CHECKPOINT_NUM}_temp${TEMPERATURE_STR}_comps${COMPLETION_LIMIT}.json.gz"
    if [ -f "${OUTPUT_PATH}.json.gz" ]; then
        echo "Output file $OUTPUT_PATH already exists, skipping. Delete to re-run."
        continue
    fi
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python $SCRIPT \
      --model $CHECKPOINT_DIR/$CHECKPOINT \
      --temperature $TEMPERATURE \
      --completion-limit $COMPLETION_LIMIT \
      --num-gpus $NUM_GPUS \
      --batch-size $BATCH_SIZE \
      --exec-batch-size $EXEC_BATCH_SIZE \
      --output $OUTPUT_PATH
done
