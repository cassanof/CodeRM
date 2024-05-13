if [ $# -eq 0 ]
  then
    echo "No arguments supplied. Please give deepspeed config file as argument"
    exit 1
fi
DS=$(realpath $1)
export WANDB_PROJECT="codeprm"
export WANDB_NAME=$(basename $0 .sh)
pushd ./finetuning-harness/
python3 -m torch.distributed.launch \
        --nproc_per_node 4 \
        main.py \
        --deepspeed="$DS" \
        --model_path="Qwen/CodeQwen1.5-7B" \
        --dataset_name="cassanof/taco_cleaned_train_codeqwen" \
        --dataset_loader="padded" \
        --mask_loss_till_token_id 92301 \
        --trim_longer \
        --no_approx_tokens \
        --output_dir="/scratch/federicoc/model_codeqwen_taco" \
        --seq_length 4096 \
        --epochs 1 \
        --fa2 \
        --batch_size 1 \
        --gradient_accumulation_steps 8 \
        --learning_rate 1e-5 \
        --num_warmup_steps 10 \
        --num_workers=$(expr $(nproc --all) - 4) \
        --no_fp16 \
        --bf16 \
        --eval_freq 0.0 \
        --perc_valid_set 0.0 \
        --save_total_limit 20
popd
