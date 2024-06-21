# LLMs for Competitive Programming

**IMPORTANT**: after cloning the repo install the submodules by running `git submodule update --init --recursive`.

Then install the package with `pip install -e .`.

#### Magic incantation for DeepSeekCoder V2

```
CUDACXX=/usr/local/cuda-12.1/bin/nvcc CMAKE_ARGS="-DLLAMA_CUBLAS=on -DCMAKE_CUDA_ARCHITECTURES=native" FORCE_CMAKE=1 pip3 install git+https://github.com/seungduk-yanolja/vllm-deepseek.git@main
```

## Evaluation

### Running the code executor

For any kind of evaluation, you first need to spin up the code executor. It's a simple sandboxed HTTP server that runs code in a Docker container. To start it, run the following command:

```
pushd ./coderm/code_exec_server
./build_and_run.sh
popd
```

If you don't want to use docker and play it dangerously, this is a working alternative (NOTE: you will need Rust installed):

```
pushd ./coderm/code_exec_server
cargo run --release & # runs it as a background process
popd
```

### Generating and executing completions

To generate completions for a benchmark, you will find a bunch of scripts in the `./coderm/eval` directory.
As an example, lets generate completions for LiveCodeBench:

```
python3 ./coderm/eval/livecodebench_eval.py --model <model-path> --output <output-results-path>
```

**IMPORTANT:** this will run the v1 of LiveCodeBench, which is what is used for the CodeRM project, but the
competitive programming model project uses v2, which can be run by using the `--dataset codegenning/livecodebench_lite_v2` argument.

By default, this will generate a single completion at `temperature=0` for each problem in the benchmark,
using no few-shots examples.
You can generate multiple completions by changing the `--completion-limit` argument and change the temperature by setting `--temperature`.
You can also add fewshots by changing the argument `--model-kind` to `few-shot`.
All the completions will be executed using the code executor, and the results will be saved in the output path.

#### Evaluating completions

After generating and executing completions, you can evaluate them using the `./coderm/eval/metrics.py` script.
As an example, to evaluate the completions generated for LiveCodeBench, you can run:

```
python3 ./coderm/eval/metrics.py --results <output-results-path>
```

The `metrics.py` script also uses a `-k` parameter for `pass@k`, if you have multiple completions for each problem.

**RECCOMENDED:** pipe your results into `column -s, -t` for a better visualization. e.g. `python3 ./coderm/eval/metrics.py --results <output-results-path> | column -s, -t`

## Training

## Dataset
