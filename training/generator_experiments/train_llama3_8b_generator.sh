#!/bin/bash
export WANDB_PROJECT="coderm"
export WANDB_NAME=$(basename $0 .sh)
export DS_SKIP_CUDA_CHECK=1
OUTDIR="./model_llama3_8b_generator"
pushd ../finetuning-harness/
# 128006: AddedToken("<|start_header_id|>"
accelerate launch main.py \
        --model_path="meta-llama/Meta-Llama-3-8B" \
        --dataset_name="codegenning/finetuning-set-llama3-v0_subset1000" \
        --dataset_loader="padded" \
        --mask_loss_till_token_id 128006 \
        --trim_longer \
        --no_approx_tokens \
        --output_dir="$OUTDIR" \
        --seq_length 4096 \
        --epochs 3 \
        --fa2 \
        --batch_size 1 \
        --gradient_accumulation_steps 4 \
        --learning_rate 1e-5 \
        --num_warmup_steps 10 \
        --num_workers=$(expr $(nproc --all) - 4) \
        --no_fp16 \
        --bf16 \
        --eval_freq 0.0 \
        --perc_valid_set 0.0 \
        --save_total_limit 20
popd
rm -fr $OUTDIR/*/*global_step*
