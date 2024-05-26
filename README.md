# CodePRM

Clone repository with submodules:

```bash
git clone --recurse-submodules git@github.com:cassanof/CodePRM
```

### Replicate 3B ORM Evaluation Results

#### 1. Install dependencies

```bash
pip install -r requirements.txt
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
