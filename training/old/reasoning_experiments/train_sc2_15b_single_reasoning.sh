if [ $# -eq 0 ]
  then
    echo "No arguments supplied. Please give deepspeed config file as argument"
    exit 1
fi
DS=$(realpath $1)
export WANDB_PROJECT="codeprm"
export WANDB_NAME=$(basename $0 .sh)
OUTDIR="./model_starcoder2_15b_taco_reasoning_single"
pushd ../finetuning-harness/
python3 -m torch.distributed.launch \
        --nproc_per_node 8 \
        main.py \
        --deepspeed="$DS" \
        --model_path="bigcode/starcoder2-15b" \
        --dataset_name="cassanof/taco_cleaned_train_sc2_reasoning_single" \
        --dataset_loader="padded" \
        --mask_loss_till_token_id 7 \
        --trim_longer \
        --no_approx_tokens \
        --output_dir="$OUTDIR" \
        --seq_length 4096 \
        --epochs 5 \
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
rm -fr $OUTDIR/*/*global_steps*
