#!/bin/bash

export WANDB_PROJECT="coderm-orm"
export WANDB_NAME="llama_3_70b_orm50_gpt4o"
pushd ../code-scorer/
python3 -m torch.distributed.launch \
  --nproc_per_node 8 \
  train.py \
  --seq_len 4096 \
  --batch_size 1 \
  --gradient_accumulation_steps 32 \
  --epochs 3 \
  --lr 1e-5 \
  --weight_decay 0.01 \
  --save_dir "./model_llama3_70b_gpt4o_orm50" \
  --dataset "codegenning/orm_dataset_50_gpt4o" \
  --eval_dataset "codegenning/orm_eval_dataset" \
  --model "codegenning/generator-llama3-70b" \
  --deepspeed "../generator_experiments/llama_3_70b_deepspeed.json" \
  --num_labels 2 \
  --bf16 \
  --no_fp16 \
  --fa2
popd
