# CodePRM

Clone repository with submodules:

```bash
git clone --recurse-submodules git@github.com:cassanof/CodePRM
```

### Replicate 3B ORM Evaluation Results

#### 1. Install dependencies and package

```bash
pip install -r requirements.txt
pip install -e .
```

#### 2. Run code execution server

```bash
pushd ./codeprm/code_exec_server;
./build_and_run.sh # runs with normal docker. see ./build_and_run_paranoid.sh to run with gVisor
# test that the server works correctly. if it doesn't throw HTTP errors, it should be fine.
# ignore errors for JavaScript and TypeScript
python3 test_reqs.py
popd
```

#### 3. Generate completions with the base generator model

```bash
mkdir ./results/3b_replication/
python3 ./codeprm/eval/livecodebench_eval.py \
    --model "codegenning/starcoder2-3b-taco-reasoning" \
    --completion-limit 100 \
    --temperature 0.8 \
    --output ./results/3b_replication/lcb_eval_base
```

#### 4. Generate scores with ORM

```bash
python3 ./codeprm/eval/run_orm.py \
    --model "codegenning/starcoder2-3b-orm-v0" \
    --input ./results/3b_replication/lcb_eval_base \
    --output ./results/3b_replication/lcb_eval_orm
```

#### 5. Check results

```bash
# split results into easy,medium,hard
python3 ./codeprm/eval/split_results_by_diff.py ./results/3b_replication/lcb_eval_orm
# check results (with pass@1)
python3 ./codeprm/eval/metrics.py ./results/3b_replication/lcb_eval_orm* | column -s, -t
# check results (with pass@100)
python3 ./codeprm/eval/metrics.py -k 100 ./results/3b_replication/lcb_eval_orm* | column -s, -t
```

## Training data generation process

1. taco uncleaned
2. clean taco
3. execution filter taco
4. generate reasoning steps
5. clean reasoning steps
6. train 15b on reasoning step data
7. use fine-tuned 15b to generate solutions on 13k taco
8. merge synthetic reasoning steps with real reasoning steps
9. clean all (using reasoning steps script)
10. train final generator on cleaned and merged reasoning steps
11. generate completions on 13k taco with final generator
12. train orm on generated completions
13. profit
