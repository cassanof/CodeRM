#!/bin/bash

GRAD_ACC=$1
LR=$2

export WANDB_PROJECT="codeprm-orm-15b-sweep"
export WANDB_NAME="sweep_starcoder2_15b_orm_raw10_${GRAD_ACC}_${LR}"
OUTDIR="./model_starcoder2_15b_orm_raw10_${GRAD_ACC}_${LR}"

pushd ../../code-scorer

python3 -m torch.distributed.launch \
  --nproc_per_node 8 \
  train.py \
  --seq_len 4096 \
  --batch_size 1 \
  --gradient_accumulation_steps $GRAD_ACC \
  --epochs 3 \
  --lr $LR \
  --weight_decay 0.01 \
  --save_dir "$OUTDIR" \
  --dataset "codegenning/orm_dataset_raw50" \
  --eval_dataset "codegenning/orm_eval_dataset" \
  --model "codegenning/generator-sc2-15b" \
  --deepspeed "./deepspeed_cfgs/no_offload.json" \
  --num_labels 2 \
  --bf16 \
  --no_fp16 \
  --fa2

# delete deepspeed checkpoints for saving space
rm -fr "$OUTDIR/*/global_step*"

popd
