python3 ../coderm/eval/livecodebench_eval.py \
  --model "codegenning/generator-sc2-15b" \
  --dataset "cassanof/livecodebench_lite_contaminated" \
  --temperature 0.8 \
  --completion-limit 100 \
  --output ./temp_generator/lcb_contaminated_generator_sc2_15b
