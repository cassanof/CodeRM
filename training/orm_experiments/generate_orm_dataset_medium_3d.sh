#!/bin/bash

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <start_idx> <end_idx; optional, defaults to max size>"
  exit 1
fi

ACTUAL_MAX_SIZE=13402

G_START_IDX=$1
G_END_IDX=$ACTUAL_MAX_SIZE
if [ "$#" -eq 2 ]; then
  G_END_IDX=$2
fi

if [ $G_START_IDX -ge $G_END_IDX ]; then
  echo "Error: Start index must be less than end index."
  exit 1
fi

pushd ../../

NUM_GPUS=8
NUM_COMPS=50
TEMPERATURE=0.8
EXEC_MULTIPLIER=2
BATCH_SIZE=512

NPROC=$(nproc)
NPROC_PER_GPU=$(((NPROC / NUM_GPUS) * EXEC_MULTIPLIER))
DATASET_SIZE=$((G_END_IDX - G_START_IDX))
MAX_ITEMS=$((DATASET_SIZE / NUM_GPUS))

echo "Computed values:"
echo "NUM_GPUS=$NUM_GPUS NPROC=$NPROC NPROC_PER_GPU=$NPROC_PER_GPU DATASET_SIZE=$DATASET_SIZE MAX_ITEMS=$MAX_ITEMS"
echo "Preset values:"
echo "NUM_COMPS=$NUM_COMPS TEMPERATURE=$TEMPERATURE BATCH_SIZE=$BATCH_SIZE"

PIDS=()
function kill_all_subprocesses() {
  for PID in "${PIDS[@]}"; do
    kill $PID
  done
}
trap kill_all_subprocesses SIGINT SIGTERM

OUTDIR="./codeprm/dataset/orm_dataset_raw${NUM_COMPS}_og_chunked_${G_START_IDX}_${G_END_IDX}/"
mkdir -p $OUTDIR

TOTAL_MAX=0
for ((i=0; i<NUM_GPUS; i++)); do
  START_IDX=$((G_START_IDX + i * MAX_ITEMS))
  if [ $i -eq $((NUM_GPUS-1)) ]; then
    MAX_ITEMS=$((G_END_IDX - START_IDX))
  fi
  TOTAL_MAX=$((TOTAL_MAX + MAX_ITEMS))
  
  echo "Processing chunk $i: start_idx=$START_IDX, max_items=$MAX_ITEMS"
  CUDA_VISIBLE_DEVICES=$i python3 codeprm/eval/taco_eval.py \
    --dataset cassanof/taco_cleaned_all \
    --split train \
    --model "codegenning/generator-sc2-15b" \
    --completion-limit $NUM_COMPS \
    --temperature $TEMPERATURE \
    --exec-batch-size $NPROC_PER_GPU \
    --batch-size $BATCH_SIZE \
    --output "$OUTDIR/chunk_$i" \
    --output-format datasets \
    --start-idx $START_IDX \
    --max-items $MAX_ITEMS &

  PIDS+=($!)
done

if [ $TOTAL_MAX -ne $DATASET_SIZE ]; then
  echo "Error: TOTAL_MAX=$TOTAL_MAX is not equal to DATASET_SIZE=$DATASET_SIZE"
  exit 1
fi

for pid in "${PIDS[@]}"; do
  wait $pid
done

popd
