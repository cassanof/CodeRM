#!/bin/bash

export WANDB_PROJECT="coderm-orm"
export WANDB_NAME="deepseekcoder-16b-linear-probe-orm50-gpt4o-epoch5"
pushd ../code-scorer/
python3 -m torch.distributed.launch \
  --nproc_per_node 8 \
  train.py \
  --seq_len 8192 \
  --batch_size 4 \
  --gradient_accumulation_steps 8 \
  --epochs 5 \
  --lr 0.03 \
  --weight_decay 0.01 \
  --save_dir "./model_deepseekcoder_16b_orm50_gpt4o_linear_probe" \
  --dataset "codegenning/orm_dataset_50_gpt4o" \
  --eval_dataset "codegenning/orm_eval_dataset" \
  --model "codegenning/deepseekcoder-16b-taco-plain800k" \
  --deepspeed "./deepspeed_cfgs/no_offload.json" \
  --num_labels 2 \
  --linear-probe \
  --bf16 \
  --no_fp16 \
  --fa2
popd
