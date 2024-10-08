#!/bin/bash

export WANDB_PROJECT="codeprm"
export WANDB_NAME="starcoder2_3b_orm_raw10"
pushd ../code-scorer/
python3 -m torch.distributed.launch \
  --nproc_per_node 8 \
  train.py \
  --seq_len 4096 \
  --batch_size 1 \
  --gradient_accumulation_steps 4 \
  --epochs 20 \
  --lr 2e-5 \
  --weight_decay 0.01 \
  --attention_dropout 0.1 \
  --save_dir "./model_starcoder2_3b_orm_raw10" \
  --dataset "cassanof/taco_orm_raw10" \
  --eval_dataset "cassanof/lcb_orm_eval" \
  --model "bigcode/starcoder2-3b" \
  --bf16 \
  --no_fp16 \
  --fa2
popd
