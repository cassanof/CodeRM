from typing import Dict, Generator, List, Tuple
from tqdm import tqdm
from pathlib import Path
from utils import gunzip_json_read, gunzip_json_write
from vllm import LLM, SamplingParams
import math
import torch
import random

FOOTER = "Is the code above correct? (Yes/No): "


def get_yesno_logprobs(out):
    logprobs = out.logprobs[0]
    yeslog = None
    nolog = None
    for l in logprobs.values():
        if l.decoded_token == "Yes":
            yeslog = l.logprob
        elif l.decoded_token == "No":
            nolog = l.logprob

    if not yeslog:
        yeslog = float("-inf")
    if not nolog:
        nolog = float("-inf")

    return yeslog, nolog


def get_yes_percentage(yes_logprob, no_logprob):
    if yes_logprob == float("-inf"):
        return 0.0
    if no_logprob == float("-inf"):
        return 1.0
    prob_yes = math.exp(yes_logprob) / \
        (math.exp(yes_logprob) + math.exp(no_logprob))
    yes_percentage = prob_yes * 100
    return yes_percentage


def trim_toplevel_py(p):
    lines = p.splitlines()
    i = len(lines)-1
    while i > 0 and lines[i].lstrip() == lines[i]:
        i -= 1
    s = "\n".join(lines[0:i+1])
    return s


def get_program_impl(p, trim_toplvl=False):
    splitpoint = "def check(candidate)"
    if "METADATA" in p:
        splitpoint = "METADATA"
    prog = p.split(splitpoint)[0].strip()
    if trim_toplvl:
        prog = trim_toplevel_py(prog)
    if '"""' not in prog:
        print(f"WARNING: Program does not have a docstring: {prog}")
    return prog + "\n"


def autodetect_dtype() -> str:
    if torch.cuda.is_bf16_supported():
        return "bfloat16"
    else:
        return "auto"


class VerifierModel:
    def verify(self, samples: List[str]) -> List[float]:
        raise NotImplementedError


class YesNoModel(VerifierModel):
    def __init__(self, name, num_gpus=1):
        self.model = LLM(
            name,
            dtype=autodetect_dtype(),
            tensor_parallel_size=num_gpus,
        )

    def verify(self, samples: List[str]) -> List[float]:
        if isinstance(samples, str):
            samples = [samples]
        samples = [sample + FOOTER for sample in samples]
        os = [o.outputs[0] for o in self.model.generate(samples, SamplingParams(
            max_tokens=1, logprobs=5, temperature=0), use_tqdm=False)]
        yesnos = [get_yesno_logprobs(o) for o in os]
        return [get_yes_percentage(yes, no) for yes, no in yesnos]


def model_factory(args):
    if args.model_kind == "logprobs":
        return YesNoModel(args.model, num_gpus=args.num_gpus)
    else:
        raise ValueError(f"Unknown model kind: {args.model_kind}")


HAS_WARNED = False


def process_results(path):
    """
    Returns a generator of: (result object, file path)
    """
    files = list(Path(path).rglob("*.results.json.gz"))
    for f in tqdm(files, total=len(files)):
        obj = gunzip_json_read(f)
        assert obj is not None, f"Failed to read {f} -- returning None"
        yield obj, f


def main(args):
    global HAS_WARNED
    random.seed(args.seed)
    model = model_factory(args)

    print(f"Verifier loaded from {args.model}")
    print(
        f"Running verifier on {args.input_dir} with samples={args.samples} and top_k={args.top_k}")

    if args.output_dir is not None:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # just report acc to stdout
    acc = 0
    iters = 0
    for obj, res_file in process_results(args.input_dir):
        res = obj["results"]
        if "tokens_info" in obj:
            tokens_info = obj["tokens_info"]
        else:
            if not HAS_WARNED:
                assert args.aggregate is None, "Cannot aggregate without tokens_info"
                print(
                    f"Warning: tokens_info not found in {res_file} -- not computing cumulative logprobs")
                HAS_WARNED = True
            tokens_info = [{} for _ in res]

        res = list(zip(res, tokens_info))
        random.shuffle(res)

        original_size = len(res)
        if args.filter_syntax_error:
            res = list(filter(lambda r: r[0]["status"] != "SyntaxError", res))

        res = res[:min(args.samples, len(res))]
        ps = [get_program_impl(r["program"], trim_toplvl=args.trim_toplvl)
              for r, _ in res]
        scores = model.verify(ps)

        # write scores to output and aggregate if needed
        for i, (score, (r, t)) in enumerate(zip(scores, res)):
            r["base_yes_prob"] = score
            if args.aggregate == "plain":
                r["yes_prob"] = score * math.exp(t["cumulative_logprob"])
            elif args.aggregate == "normalized":
                r["yes_prob"] = score * \
                    math.exp(t["cumulative_logprob"] / t["len"])
            else:
                r["yes_prob"] = score

            # get rid of tokens_info. ciao ciao
            res[i] = r

        # check scores for correctness
        best_statuses = [None] * args.top_k
        best_scores = [-1] * args.top_k
        best_p = [None] * args.top_k
        for r, p in zip(res, ps):
            assert isinstance(r, dict)
            if p.strip() == "":
                continue  # skip empty programs

            current_score = r["yes_prob"]
            smallest_idx = min(range(args.top_k), key=lambda i: best_scores[i])
            if current_score > best_scores[smallest_idx]:
                best_scores[smallest_idx] = current_score
                best_statuses[smallest_idx] = r["status"]
                best_p[smallest_idx] = p

        obj["best"] = {
            "statuses": best_statuses,
            "scores": best_scores,
            "programs": best_p
        }

        obj["results"] = res
        obj["samples"] = args.samples
        obj["top_k"] = args.top_k
        obj["model_kind"] = args.model_kind
        obj["aggregate"] = args.aggregate
        obj["original_size"] = original_size
        if args.output_dir is not None:
            out_file = Path(args.output_dir) / res_file.name
            gunzip_json_write(out_file, obj)

        if args.verbose == 1:
            # just the best score list
            print(f" *** {iters} ***")
            print(f" Scores: {best_scores}")

        if args.verbose == 2:
            print("################### BEST P #####################")
            for i in range(args.top_k):
                print(f" *** {i+1} ***")
                print(f" Score: {best_scores[i]}")
                print(f" Status: {best_statuses[i]}")
                print(best_p[i])
            print("################### END BEST P #####################")

        if any([status == "OK" for status in best_statuses]):
            acc += 1
        iters += 1

    print(f"Accuracy: {acc / iters}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Directory containing the MultiPL-E result files to verify")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to write the verified results to")
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--aggregate", type=str, default=None,
                        choices=[None, "plain", "normalized"])
    parser.add_argument("--samples", type=int, default=100,
                        help="Number of samples to consider in each solution-space")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-kind", type=str, default="logprobs",
                        choices=["logprobs"])
    parser.add_argument("--no-filter-syntax-error", dest="filter_syntax_error",
                        action="store_false", help="Do not filter out syntax errors")
    parser.add_argument("--verbose", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--trim-toplvl", action="store_true")
    args = parser.parse_args()
    main(args)
