python3 coderm/eval/livecodebench_eval.py --model cassanof/starcoder2-3b-taco-reasoning --output ./results/greedy_reasonings_bootstrap/lcb_sc2_reasoning_bootstrap_3b_greedy;
python3 coderm/eval/livecodebench_eval.py --model cassanof/starcoder2-7b-taco-reasoning --output ./results/greedy_reasonings_bootstrap/lcb_sc2_reasoning_bootstrap_7b_greedy;
python3 coderm/eval/livecodebench_eval.py --model cassanof/starcoder2-15b-taco-reasoning --output ./results/greedy_reasonings_bootstrap/lcb_sc2_reasoning_bootstrap_15b_greedy;
python3 coderm/eval/livecodebench_eval.py --model cassanof/starcoder2-3b-taco-reasoning --temperature 0.8 --completion-limit 100 --output ./results/temp_reasonings_bootstrap/lcb_sc2_reasoning_bootstrap_3b_temp08;
python3 coderm/eval/livecodebench_eval.py --model cassanof/starcoder2-7b-taco-reasoning --temperature 0.8 --completion-limit 100 --output ./results/temp_reasonings_bootstrap/lcb_sc2_reasoning_bootstrap_7b_temp08;
python3 coderm/eval/livecodebench_eval.py --model cassanof/starcoder2-15b-taco-reasoning --temperature 0.8 --completion-limit 100 --output ./results/temp_reasonings_bootstrap/lcb_sc2_reasoning_bootstrap_15b_temp08;
