import datasets
from codeprm.utils import chunkify, container_restart
from tqdm import tqdm
import json
from codeprm.execution import smart_exec_tests
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, default="./taco_cleaned")
parser.add_argument("--executor", type=str, default="http://127.0.0.1:8000")
parser.add_argument("--container_name", type=str, default="code-exec")
parser.add_argument("--max-solns", type=int, default=75)
parser.add_argument("--max-attempts", type=int, default=100)
parser.add_argument("--workers", type=int, default=os.cpu_count() - 1)
parser.add_argument("--batch-size", type=int, default=os.cpu_count() * 2)
parser.add_argument("--sample", type=int, default=None)
args = parser.parse_args()

ds = datasets.load_from_disk("./taco_cleaned")

if args.sample is not None:
    ds = ds.select(range(args.sample))

ds = ds.to_list()
print("Loaded dataset: ", len(ds))
chunks = chunkify(ds, args.batch_size)

dses = []
for chunk in chunks:
    dses.append(datasets.Dataset.from_list(chunk))

print(f"Loaded dataset chunks (batch size: {args.batch_size}): {len(dses)}")


def filter_not_executing(ex):
    passing_solns = []
    for i, sol in enumerate(ex["solutions"]):
        if len(passing_solns) >= args.max_solns:
            break
        if i >= args.max_attempts:
            break
        passing, e = smart_exec_tests(
            sol, json.loads(ex["input_output"]), executor=args.executor, timeout=90)
        if passing:
            passing_solns.append(sol)
        else:
            print("\n".join(e.split("\n")[:10]))

    return {
        "solutions": passing_solns,
    }


final_ds = []

for ds in tqdm(dses, desc="Filtering all datasets"):
    ds = ds.map(filter_not_executing, num_proc=args.workers)
    ds = ds.filter(lambda x: len(x["solutions"]) > 0)
    final_ds.extend(ds.to_list())
    # restart docker container
    print("Done with a batch, restarting docker container for stability...")
    container_restart(name=args.container_name)

final_ds = datasets.Dataset.from_list(final_ds)
final_ds.save_to_disk(args.input_dir + "_exec_filtered")
