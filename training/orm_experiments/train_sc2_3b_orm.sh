python3 -m torch.distributed.launch \
  --nproc_per_node 8 \
  train.py \
  --seq_len 4096 \
  --batch_size 4 \
  --gradient_accumulation_steps 1 \
  --epochs 10 \
  --lr 2e-5 \
  --weight_decay 0.01 \
  --save_dir "./model_starcoder2_3b_orm_raw10" \
  --dataset "cassanof/taco_orm_raw10" \
  --eval_dataset "cassanof/lcb_orm_eval" \
  --model "bigcode/starcoder2-3b" \
  --bf16 \
  --fa2 \
  --no_fp16
