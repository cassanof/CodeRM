#!/bin/bash

export WANDB_PROJECT="coderm-orm"
export WANDB_NAME="starcoder2_7b_orm50_on_top_of_linear_probe"
pushd ../code-scorer/
python3 -m torch.distributed.launch \
  --nproc_per_node 4 \
  train.py \
  --seq_len 4096 \
  --batch_size 1 \
  --gradient_accumulation_steps 32 \
  --epochs 3 \
  --lr 1e-5 \
  --weight_decay 0.01 \
  --save_dir "./model_starcoder2_7b_orm50" \
  --dataset "codegenning/orm_dataset_raw50" \
  --eval_dataset "codegenning/orm_eval_dataset" \
  --model "./model_starcoder2_7b_orm50_linear_probe/checkpoint-1782/" \
  --deepspeed "./deepspeed_cfgs/no_offload.json" \
  --num_labels 2 \
  --bf16 \
  --no_fp16 \
  --fa2
popd
