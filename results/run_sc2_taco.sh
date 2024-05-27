python3 codeprm/eval/livecodebench_eval.py --model cassanof/starcoder2-3b-taco --output ./results/greedy_finetunes/lcb_sc2_taco_3b_greedy;
python3 codeprm/eval/livecodebench_eval.py --model cassanof/starcoder2-7b-taco --output ./results/greedy_finetunes/lcb_sc2_taco_7b_greedy;
python3 codeprm/eval/livecodebench_eval.py --model cassanof/starcoder2-15b-taco --output ./results/greedy_finetunes/lcb_sc2_taco_15b_greedy;
python3 codeprm/eval/livecodebench_eval.py --model cassanof/starcoder2-3b-taco --temperature 0.8 --completion-limit 100 --output ./results/temp_finetunes/lcb_sc2_taco_3b_temp08;
python3 codeprm/eval/livecodebench_eval.py --model cassanof/starcoder2-7b-taco --temperature 0.8 --completion-limit 100 --output ./results/temp_finetunes/lcb_sc2_taco_7b_temp08;
python3 codeprm/eval/livecodebench_eval.py --model cassanof/starcoder2-15b-taco --temperature 0.8 --completion-limit 100 --output ./results/temp_finetunes/lcb_sc2_taco_15b_temp08;
