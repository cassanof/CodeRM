import datasets
import json
from codeprm.execution import smart_exec_tests
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, default="./taco_cleaned")
parser.add_argument("--executor", type=str, default="http://127.0.0.1:8000")
parser.add_argument("--max-solns", type=int, default=20)
parser.add_argument("--max-attempts", type=int, default=40)
parser.add_argument("--workers", type=int, default=1)
parser.add_argument("--sample", type=int, default=None)
args = parser.parse_args()

ds = datasets.load_from_disk("./taco_cleaned")

if args.sample is not None:
    ds = ds.select(range(args.sample))


def filter_not_executing(ex):
    passing_solns = []
    for i, sol in enumerate(ex["solutions"]):
        if len(passing_solns) >= args.max_solns:
            break
        if i >= args.max_attempts:
            break
        retries = 0
        while retries < 3:
            retries += 1
            passing, e = smart_exec_tests(
                sol, json.loads(ex["input_output"]), executor=args.executor, timeout=90)
            if passing:
                passing_solns.append(sol)
            else:
                if "BrokenPipeError" in e:
                    print("Retrying due to BrokenPipeError")
                    continue
                print("\n".join(e.split("\n")[:10]))
            break

    return {
        "solutions": passing_solns,
    }


ds = ds.map(filter_not_executing, num_proc=args.workers)
ds = ds.filter(lambda x: len(x["solutions"]) > 0)

ds.save_to_disk(args.input_dir + "_exec_filtered")
