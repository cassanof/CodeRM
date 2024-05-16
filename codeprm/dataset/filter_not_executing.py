import datasets
from codeprm.utils import chunkify
from tqdm import tqdm
import json
from codeprm.execution import parse_time_limit, smart_exec_tests
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, default="./taco_cleaned")
parser.add_argument("--executor", type=str, default="http://127.0.0.1:8000")
parser.add_argument("--container_name", type=str, default="code-exec")
parser.add_argument("--max-solns", type=int, default=75)
parser.add_argument("--max-attempts", type=int, default=100)
parser.add_argument("--workers", type=int, default=os.cpu_count() - 1)
parser.add_argument("--timeout", type=int, default=30)
parser.add_argument("--sample", type=int, default=None)
args = parser.parse_args()

ds = datasets.load_from_disk("./taco_cleaned")

if args.sample is not None:
    ds = ds.select(range(args.sample))


def filter_not_executing(ex):
    time_limit = parse_time_limit(ex["time_limit"], default=args.timeout)
    passing_solns = []
    for i, sol in enumerate(ex["solutions"]):
        if len(passing_solns) >= args.max_solns:
            break
        if i >= args.max_attempts:
            break
        passing, e = smart_exec_tests(
            sol, json.loads(ex["input_output"]), executor=args.executor, timeout=time_limit)
        if passing:
            passing_solns.append(sol)
        else:
            print("\n".join(e.split("\n")[:10]))

    return {
        "solutions": passing_solns,
    }


ds = ds.map(filter_not_executing, num_proc=args.workers, load_from_cache_file=False)
ds = ds.filter(lambda x: len(x["solutions"]) > 0)

ds.save_to_disk(args.input_dir + "_exec_filtered")
