python3 coderm/eval/livecodebench_eval.py --model Qwen/CodeQwen1.5-7B --model-kind few-shot --output ./results/codeqwen_rerun/lcb_codeqwen_7b_greedy;
python3 coderm/eval/livecodebench_eval.py --model Qwen/CodeQwen1.5-7B --model-kind few-shot --temperature 0.8 --completion-limit 100 --output ./results/codeqwen_rerun/lcb_codeqwen_7b_temp08;
