BASE_MODEL="codegenning/generator-sc2-15b"
REWARD_MODEL="codegenning/orm-sc2-15b-v0"
OUTDIR="./sc2_15b_rl"

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
    --local_rollout_forward_batch_size 1 \
    --response_length 4096 \
    --non_eos_penalty \
    --fp16 \
    --stop_token eos \
    --num_labels 2 \
    --kl_coef 0.03 \
    --logging_steps 1 \
    --num_sample_generation 5
