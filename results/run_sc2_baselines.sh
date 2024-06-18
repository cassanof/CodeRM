python3 coderm/eval/livecodebench_eval.py --model bigcode/starcoder2-3b --model-kind few-shot --output ./results/greedy_baselines/lcb_sc2_3b_greedy;
python3 coderm/eval/livecodebench_eval.py --model bigcode/starcoder2-7b --model-kind few-shot --output ./results/greedy_baselines/lcb_sc2_7b_greedy;
python3 coderm/eval/livecodebench_eval.py --model bigcode/starcoder2-15b --model-kind few-shot --output ./results/greedy_baselines/lcb_sc2_15b_greedy;
python3 coderm/eval/livecodebench_eval.py --model bigcode/starcoder2-3b --model-kind few-shot --temperature 0.8 --completion-limit 100 --output ./results/temp_baselines/lcb_sc_3b_temp08;
python3 coderm/eval/livecodebench_eval.py --model bigcode/starcoder2-7b --model-kind few-shot --temperature 0.8 --completion-limit 100 --output ./results/temp_baselines/lcb_sc_7b_temp08;
python3 coderm/eval/livecodebench_eval.py --model bigcode/starcoder2-15b --model-kind few-shot --temperature 0.8 --completion-limit 100 --output ./results/temp_baselines/lcb_sc_15b_temp08;
