python3 coderm/eval/livecodebench_eval.py --model meta-llama/Meta-Llama-3-8B --model-kind few-shot --output ./results/greedy_baselines/lcb_llama3_8b_greedy;
python3 coderm/eval/livecodebench_eval.py --model meta-llama/Meta-Llama-3-70B --model-kind few-shot --output ./results/greedy_baselines/lcb_llama3_70b_greedy --num-gpus 4;
python3 coderm/eval/livecodebench_eval.py --model meta-llama/Meta-Llama-3-8B --model-kind few-shot --temperature 0.8 --completion-limit 100 --output  ./results/temp_baselines/lcb_llama3_8b_temp08;
python3 coderm/eval/livecodebench_eval.py --model meta-llama/Meta-Llama-3-70B --model-kind few-shot --temperature 0.8 --completion-limit 100 --output ./results/temp_baselines/lcb_llama3_70b_temp08 --num-gpus 4;
