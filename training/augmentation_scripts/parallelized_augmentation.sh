#!/bin/bash
pushd ../../
NUM_GPUS=8
DATASET_SIZE=7187
MAX_ITEMS=$((DATASET_SIZE / NUM_GPUS))
MAX_SOLNS=5000

PIDS=()
function kill_all_subprocesses() {
  for PID in "${PIDS[@]}"; do
    kill $PID
  done
}
trap kill_all_subprocesses SIGINT SIGTERM

OUTDIR="./coderm/dataset/on_disk/augmented_taco_chunked/"
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
  CUDA_VISIBLE_DEVICES=$i python3 coderm/dataset/augment_with_resoning_steps.py \
    --batch-size 1024 \
    --output "$OUTDIR/chunk_$i" \
    --output-format datasets \
    --max-solns $MAX_SOLNS \
    --retry-k 3 \
    --start-idx $START_IDX \
    --sample $MAX_ITEMS &

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
