BASE_MODEL="codegenning/generator-sc2-7b"
REWARD_MODEL="/mnt/efs/federicocassano/codeprm/training/code-scorer/model_starcoder2_7b_orm50_mse"
OUTDIR="./sc2_7b_rl_testing"

CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file ./z2.yaml --gpu_ids 4,5,6,7 --num_processes 4 train_rloo.py \
    --output_dir $OUTDIR \
    --num_ppo_epochs 2 \
    --num_mini_batches 2 \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --total_episodes 100_000 \
    --model_name_or_path $BASE_MODEL \
    --sft_model_path $BASE_MODEL \
    --reward_model_path $REWARD_MODEL \
    --local_rollout_forward_batch_size 16 \
    --response_length 2048 \
    --non_eos_penalty \
    --stop_token eos \
    --kl_coef 0.03 \
    --logging_steps 1 \
    --num_sample_generation 0 \
    --logging_first_step False
