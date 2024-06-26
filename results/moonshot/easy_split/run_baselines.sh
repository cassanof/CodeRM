
python3 coderm/eval/livecodebench_eval.py --model "mistralai/Codestral-22B-v0.1" --output ./results/moonshot/easy_split/easy_codestral22b_greedy.json --model-kind few-shot-chat --dataset codegenning/livecodebench_lite_contaminated_easy_v2
python3 coderm/eval/livecodebench_eval.py --model "codegenning/DeepSeek-Coder-V2-Lite-Base" --output ./results/moonshot/easy_split/easy_deepseeklite_base_greedy.json --model-kind few-shot --dataset codegenning/livecodebench_lite_contaminated_easy_v2
python3 coderm/eval/livecodebench_eval.py --model "codegenning/DeepSeek-Coder-V2-Lite-Instruct" --output ./results/moonshot/easy_split/easy_deepseeklite_instruct_greedy.json --model-kind few-shot-chat --dataset codegenning/livecodebench_lite_contaminated_easy_v2
python3 coderm/eval/livecodebench_eval.py --model "meta-llama/Meta-Llama-3-8B" --output ./results/moonshot/easy_split/easy_llama3_8b_greedy.json --model-kind few-shot --dataset codegenning/livecodebench_lite_contaminated_easy_v2
python3 coderm/eval/livecodebench_eval.py --model "meta-llama/Meta-Llama-3-8B-Instruct" --output ./results/moonshot/easy_split/easy_llama3_8b_instruct_greedy.json --model-kind few-shot-chat --dataset codegenning/livecodebench_lite_contaminated_easy_v2

