#!/bin/bash

export WANDB_PROJECT="codeprm-orm"
export WANDB_NAME="starcoder2_7b_orm50_linear_probe_epoch5"
pushd ../code-scorer/
python3 -m torch.distributed.launch \
  --nproc_per_node 4 \
  train.py \
  --seq_len 4096 \
  --batch_size 1 \
  --gradient_accumulation_steps 32 \
  --epochs 5 \
  --lr 0.03 \
  --weight_decay 0.01 \
  --save_dir "./model_starcoder2_7b_orm50_linear_probe" \
  --dataset "codegenning/orm_dataset_raw50" \
  --eval_dataset "codegenning/orm_eval_dataset" \
  --model "/mnt/efs/federicocassano/codeprm/training/finetuning-harness/model_starcoder2_7b_generator/checkpoint-1068" \
  --deepspeed "./deepspeed_cfgs/no_offload.json" \
  --num_labels 2 \
  --linear-probe \
  --bf16 \
  --no_fp16 \
  --fa2
popd
