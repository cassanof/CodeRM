#!/bin/bash

SINGLE_GPU="1"
MULTI_GPU="3,4,5,6" # for llama3-70b

RMS=(
  "codegenning/orm-llama3-70b-v0"
  "codegenning/orm-sc2-15b-v0"
  "codegenning/orm-sc2-7b-v0"
  "codegenning/orm-sc2-3b-v0"
)

INPUT_FILES=(
  "results/temp_generator/lcb_generator_sc2_3b_temp08.json.gz"
  "results/temp_generator/lcb_generator_sc2_7b_temp08.json.gz"
  "results/temp_generator/lcb_generator_sc2_15b_temp08.json.gz"
  "results/temp_generator/lcb_generator_llama3_70b_temp08.json.gz"
)


pushd $(dirname $0)
pushd ../

OUTDIR="./results/rm_grid"
mkdir -p $OUTDIR

for rm in "${RMS[@]}"; do
  for input_file in "${INPUT_FILES[@]}"; do
    # input without .json.gz
    in_trim=${input_file%.json.gz}
    rm_base=${rm#codegenning/orm-}
    outfile="$OUTDIR/${in_trim}_${rm_base}.json.gz"
    echo "Running RM $rm on input file $input_file - writing to $outfile"
    if [ "$rm" == "codegenning/orm-llama3-70b-v0" ]; then
      echo "running llama model"
      CUDA_VISIBLE_DEVICES=$MULTI_GPU python3 ./coderm/eval/run_orm.py --model "$rm" --input "$input_file" --output "$outfile" --device "auto"
    else
      echo "running starcoder model"
      CUDA_VISIBLE_DEVICES=$SINGLE_GPU python3 ./coderm/eval/run_orm.py --model "$rm" --input "$input_file" --output "$outfile"
    fi
  done
done
popd
popd