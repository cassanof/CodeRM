BASE_MODEL="codegenning/generator-sc2-7b"
# TODO: switch out to BCE model once code is good
REWARD_MODEL="codegenning/orm-sc2-7b-mse-v0"
OUTDIR="./sc2_7b_rl_testing"

accelerate launch --config_file ./z3.yaml --num_processes 8 train_rloo.py \
    --output_dir $OUTDIR \
    --num_ppo_epochs 2 \
    --num_mini_batches 2 \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --total_episodes 25_000 \
    --model_name_or_path $BASE_MODEL \
    --sft_model_path $BASE_MODEL \
    --reward_model_path $REWARD_MODEL \
    --local_rollout_forward_batch_size 4 \
    --response_length 2048 \
    --non_eos_penalty \
    --fp16 \
    --stop_token eos \
    --kl_coef 0.03 \
    --logging_steps 1 \
    --num_sample_generation 5
