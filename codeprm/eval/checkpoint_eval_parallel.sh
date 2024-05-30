#!/bin/bash
if [ "$#" -ne 2 ]; then
    echo "Usage: checkpoint_eval.sh <script> <checkpoints dir> <num gpus>"
    exit 1
fi

SCRIPT=$1
CHECKPOINT_DIR=$2
NUM_GPUS=$3

# ENV vars with default values
TEMPERATURE=${TEMPERATURE:-0.0}
COMPLETION_LIMIT=${COMPLETION_LIMIT:-1}
BATCH_SIZE=${BATCH_SIZE:-256}
EXEC_BATCH_SIZE=${EXEC_BATCH_SIZE:-$(nproc)}

# string version of TEMPERATURE, without the dot
TEMPERATURE_STR=$(echo $TEMPERATURE | tr -d . | tr - _)

SCRIPT_NAME=$(basename $SCRIPT)
SCRIPT_NAME=${SCRIPT_NAME%.*}

CHECKPOINT_DIRS=()
for dir in $CHECKPOINT_DIR/*; do
    CHECKPOINT_DIRS+=($dir)
done

CHECKPOINT_GROUPS=()
temp_group=""
for (( i=0; i<${#CHECKPOINT_DIRS[@]}; i++ )); do
    if (( i % NUM_GPUS == 0 && i != 0 )); then
        # remove the trailing '|' from the current group before adding to CHECKPOINT_GROUPS
        temp_group=${temp_group%|}
        CHECKPOINT_GROUPS+=("$temp_group")
        temp_group="${CHECKPOINT_DIRS[$i]}|"
    else
        temp_group+="${CHECKPOINT_DIRS[$i]}|"
    fi
done

# add the last temp_group to CHECKPOINT_GROUPS, removing the trailing '|' if present
if [[ ! -z "$temp_group" ]]; then
    temp_group=${temp_group%|}
    CHECKPOINT_GROUPS+=("$temp_group")
fi

for (( gi=0; gi<${#CHECKPOINT_GROUPS[@]}; gi++ )); do
  IFS='|' read -ra ADDR <<< "${CHECKPOINT_GROUPS[$gi]}"
  echo "Firing off gpu-group $gi"
  PIDS=()

  for (( i=0; i<${#ADDR[@]}; i++ ));
  do
      CHECKPOINT=${ADDR[$i]}
      BASEDIR=$(basename $CHECKPOINT)
      OUTPUT_PATH="${CHECKPOINT}/${SCRIPT_NAME}_${BASEDIR}_temp${TEMPERATURE_STR}_comps${COMPLETION_LIMIT}"
      if [ -f "${OUTPUT_PATH}.json.gz" ]; then
          echo "Output file $OUTPUT_PATH.json.gz already exists, skipping. Delete to re-run."
          continue
      fi
      echo "Starting process $i with checkpoint ${ADDR[$i]} - output: $OUTPUT_PATH"
      ID=$i
      CUDA_VISIBLE_DEVICES=$ID python $SCRIPT \
        --model $CHECKPOINT \
        --temperature $TEMPERATURE \
        --completion-limit $COMPLETION_LIMIT \
        --batch-size $BATCH_SIZE \
        --exec-batch-size $EXEC_BATCH_SIZE \
        --output $OUTPUT_PATH &
      PIDS+=($!)
  done

  echo "Waiting for all processes to finish... Pids: ${PIDS[@]}"

  # capture a ctrl-c and kill all processes
  function ctrl_c() {
      echo "Trapped CTRL-C, killing all processes..."
      for pid in ${PIDS[@]}; do
          kill $pid
      done
      exit 1
  }

  trap ctrl_c INT

  wait # wait for all background processes to finish
done
