#!/bin/bash

export WANDB_PROJECT="coderm-orm"
export WANDB_NAME="llama3-70b-linear-probe-orm50-epoch5"
pushd ../code-scorer/
python3 -m torch.distributed.launch \
  --nproc_per_node 8 \
  train.py \
  --seq_len 8192 \
  --batch_size 1 \
  --gradient_accumulation_steps 32 \
  --epochs 5 \
  --lr 0.03 \
  --weight_decay 0.01 \
  --save_dir "./model_starcoder2_7b_orm50_linear_probe_v2" \
  --dataset "codegenning/orm_dataset_raw50_dedup" \
  --eval_dataset "codegenning/orm_eval_dataset" \
  --model "codegenning/generator-llama3-70b" \
  --deepspeed "./deepspeed_cfgs/no_offload.json" \
  --num_labels 2 \
  --linear-probe \
  --bf16 \
  --no_fp16 \
  --fa2
popd
