#!/bin/bash
pushd ../../
python3 codeprm/eval/taco_eval.py \
  --dataset cassanof/taco_cleaned_all \
  --split train \
  --model cassanof/starcoder2-15b-taco-reasoning \
  --num-gpus 8 \
  --batch-size 4096 \
  --completion-limit 100 \
  --temperature 0.8 \
  --output ./codeprm/dataset/cassanof/orm_dataset_raw100_og \
  --output-format datasets
popd
