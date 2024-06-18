if [ "$#" -ne 1 ]; then
  echo "Usage: $0 TEMPERATURE"
  exit 1
fi
TEMP=$1
mkdir -p ./temperature_experiment
python3 ../coderm/eval/livecodebench_eval.py \
  --model "codegenning/generator-sc2-15b" \
  --temperature $TEMP \
  --completion-limit 250 \
  --output ./temperature_experiment/lcb_sc2_15b_generator_temp${TEMP};
