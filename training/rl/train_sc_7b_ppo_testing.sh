BASE_MODEL=${1:-"codegenning/generator-sc2-7b"}
# TODO: switch out to BCE model once code is good
REWARD_MODEL=${2:-"codegenning/orm-sc2-7b-bce-v0"}
OUTDIR="./sc2_7b_rl_testing"

set -x 

deepspeed ./train_code_ppo.py     \
    --pretrain $BASE_MODEL \
    --reward_pretrain $REWARD_MODEL \
    --save_path $OUTDIR \
    --save_steps -1 \
    --logging_steps 1 \
    --eval_steps 200 \
    --micro_train_batch_size 2 \
    --train_batch_size 64 \
    --micro_rollout_batch_size 4 \
    --rollout_batch_size 1024 \
    --max_epochs 1 \
    --prompt_max_len 2048 \
    --generate_max_len 4096 \
    --zero_stage 2 \
    --bf16 \
    --actor_learning_rate 5e-7 \
    --critic_learning_rate 9e-6 \
    --init_kl_coef 0.01 \
    --prompt_data "codegenning/taco-rl" \
    --prompt_data_probs 1.0 \
    --max_samples 80000 \
    --normalize_reward \
    --actor_init_on_gpu \
    --adam_offload \
    --flash_attn \
    --gradient_checkpointing \
    --use_wandb True \
    --input_template="" \
    --input_key="prompt"
