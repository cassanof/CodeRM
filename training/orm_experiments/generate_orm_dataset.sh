#!/bin/bash
pushd ../../
NUM_GPUS=8
NPROC=$(nproc)
NPROC_PER_GPU=$((NPROC / NUM_GPUS))
DATASET_SIZE=13402
MAX_ITEMS=$((DATASET_SIZE / NUM_GPUS))

PIDS=()
function kill_all_subprocesses() {
  for PID in "${PIDS[@]}"; do
    kill $PID
  done
}
trap kill_all_subprocesses SIGINT SIGTERM

OUTDIR="./codeprm/dataset/orm_dataset_raw100_og_chunked/"
mkdir $OUTDIR

TOTAL_MAX=0
for ((i=0; i<NUM_GPUS; i++)); do
  START_IDX=$((i * MAX_ITEMS))
  # Handle the last chunk if dataset size is not divisible by NUM_GPUS
  if [ $i -eq $((NUM_GPUS-1)) ]; then
    MAX_ITEMS=$((DATASET_SIZE - START_IDX))
  fi
  TOTAL_MAX=$((TOTAL_MAX + MAX_ITEMS))
  
  echo "Processing chunk $i: start_idx=$START_IDX, max_items=$MAX_ITEMS"
  python3 codeprm/eval/taco_eval.py \
    --dataset cassanof/taco_cleaned_all \
    --split train \
    --model cassanof/starcoder2-15b-taco-reasoning \
    --completion-limit 100 \
    --temperature 0.8 \
    --exec_batch_size $NPROC_PER_GPU \
    --output "$OUTDIR/chunk_$i" \
    --output-format datasets \
    --start_idx $START_IDX \
    --max_items $MAX_ITEMS &

  # Add the process ID to the PIDS array
  PIDS+=($!)
done

# make sure TOTAL_MAX is equal to DATASET_SIZE. just a sanity check
if [ $TOTAL_MAX -ne $DATASET_SIZE ]; then
  echo "Error: TOTAL_MAX=$TOTAL_MAX is not equal to DATASET_SIZE=$DATASET_SIZE"
  exit 1
fi

# Wait for all processes to complete
for pid in "${PIDS[@]}"; do
  wait $pid
done

popd
