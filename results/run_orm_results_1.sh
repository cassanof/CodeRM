ORM_7B="./training/code-scorer/model_starcoder2_7b_orm50/checkpoint-1189"
ORM_7B_NR="./training/code-scorer/model_starcoder2_7b_orm50_no_reasoning/checkpoint-1782"
ORM_7B_MUT="./training/code-scorer/model_starcoder2_7b_orm_mutated/checkpoint-2328"

RES_7B="./results/temp_generator/lcb_generator_sc2_7b_temp08.json.gz"
RES_15B="./results/temp_generator/lcb_generator_sc2_15b_temp08.json.gz"

# sc7b 7b orm
CUDA_VISIBLE_DEVICES="0" python3 codeprm/eval/run_orm.py --model $ORM_7B --input $RES_7B --output ./results/orm_results/lcb_generator_sc2_7b_temp08_orm_7b.json.gz &

# sc7b 7b orm no reasoning
CUDA_VISIBLE_DEVICES="1" python3 codeprm/eval/run_orm.py --model $ORM_7B_NR --input $RES_7B --strip-comments --output ./results/reasoning_ablations/lcb_generator_sc2_7b_temp08_orm_7b_no_reasoning.json.gz &

# sc15b 7b orm no reasoning
CUDA_VISIBLE_DEVICES="3" python3 codeprm/eval/run_orm.py --model $ORM_7B_NR --input $RES_15B --strip-comments --output ./results/reasoning_ablations/lcb_generator_sc2_15b_temp08_orm_7b_no_reasoning.json.gz &

# sc7b 7b orm mutated
CUDA_VISIBLE_DEVICES="4" python3 codeprm/eval/run_orm.py --model $ORM_7B_MUT --input $RES_7B --output ./results/mutation_ablations/lcb_generator_sc2_7b_temp08_orm_7b_mutated.json.gz &

# sc15b 7b orm mutated
CUDA_VISIBLE_DEVICES="5" python3 codeprm/eval/run_orm.py --model $ORM_7B_MUT --input $RES_15B --output ./results/mutation_ablations/lcb_generator_sc2_15b_temp08_orm_7b_mutated.json.gz &
