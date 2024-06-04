#!/bin/bash

export WANDB_PROJECT="codeprm-orm"
export WANDB_NAME="starcoder2_7b_orm50_no_reasoning"
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
  --save_dir "./model_starcoder2_7b_orm50_no_reasoning" \
  --dataset "codegenning/orm_dataset_raw50_no_reasoning" \
  --eval_dataset "codegenning/orm_eval_dataset" \
  --model "<MODEL HERE>" \
  --deepspeed "./deepspeed_cfgs/no_offload.json" \
  --num_labels 2 \
  --bf16 \
  --no_fp16 \
  --fa2
popd
