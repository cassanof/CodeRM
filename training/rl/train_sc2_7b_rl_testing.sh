BASE_MODEL="/mnt/efs/federicocassano/codeprm/training/finetuning-harness/model_starcoder2_7b_generator/checkpoint-1068"
# TODO: switch out to BCE model once code is good
REWARD_MODEL="/mnt/efs/federicocassano/codeprm/training/code-scorer/model_starcoder2_7b_orm50_mse/checkpoint-1782"
OUTDIR="./sc2_7b_rl_testing"

CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file ./z3.yaml --gpu_ids 4,5,6,7 --num_processes 4 train_rloo.py \
    --output_dir $OUTDIR \
    --num_ppo_epochs 2 \
    --num_mini_batches 2 \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --total_episodes 100_000 \
    --model_name_or_path $BASE_MODEL \
    --sft_model_path $BASE_MODEL \
    --reward_model_path $REWARD_MODEL \
    --local_rollout_forward_batch_size 4 \
    --response_length 1028 \
    --non_eos_penalty \
    --fp16 \
    --stop_token eos \
    --kl_coef 0.03 \
    --logging_steps 1 \
    --num_sample_generation 0 \
    --logging_first_step False
