TEMPS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
mkdir ./temperature_experiment
for TEMP in ${TEMPS[@]}; do
  python3 codeprm/eval/livecodebench_eval.py \
    --model "codegenning/generator-sc2-15b" \
    --temperature $TEMP \
    --completion-limit 250 \
    --num-gpus 4 \
    --output ./temperature_experiment/lcb_sc2_15b_generator_temp${TEMP};
done
