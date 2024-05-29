#!/bin/bash

export WANDB_PROJECT="codeprm"
export WANDB_NAME="starcoder2_7b_orm_raw10_bce_reasoning"
pushd ../code-scorer/
python3 -m torch.distributed.launch \
  --nproc_per_node 8 \
  train.py \
  --seq_len 4096 \
  --batch_size 1 \
  --gradient_accumulation_steps 4 \
  --epochs 3 \
  --lr 2e-5 \
  --weight_decay 0.01 \
  --save_dir "./model_starcoder2_7b_orm_raw10_bce_epoch3" \
  --dataset "cassanof/taco_orm_raw10" \
  --eval_dataset "cassanof/lcb_orm_eval" \
  --model "cassanof/starcoder2-7b-taco-reasoning" \
  --deepspeed "./deepspeed_cfgs/no_offload.json" \
  --num_labels 2 \
  --bf16 \
  --no_fp16 \
  --fa2
popd
