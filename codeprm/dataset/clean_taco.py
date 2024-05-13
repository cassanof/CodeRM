import datasets
import numpy as np
import os
import json
import sys
import ast
if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)


from transformers import AutoTokenizer


def does_parse(code):
    try:
        ast.parse(code)
        return True
    except:
        return False


def patch_tests(ex):  # patch up solutions tests
    tests = json.loads(ex["input_output"])

    if "fn_name" in tests:
        outputs = []
        for o in tests["outputs"]:
            outputs.append(o[0])
        tests["outputs"] = outputs
        # if inputs are not a list, then make it a single element list
        inputs = []
        for i in tests["inputs"]:
            if not isinstance(i, list):
                i = [i]
            inputs.append(i)
        tests["inputs"] = inputs
    else:  # io tests
        inputs = []
        for i in tests["inputs"]:
            if isinstance(i, list):
                i = "\n".join(map(str, i))
            inputs.append(i)

        tests["inputs"] = inputs

    return {"input_output": json.dumps(tests)}


def main(args):
    tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder2-15b")

    ds = datasets.load_dataset(
        "BAAI/TACO", split=args.split, trust_remote_code=True)
    # json load the solutions
    ds = ds.map(lambda x: {"solutions": json.loads(
        x["solutions"])}, num_proc=os.cpu_count())
    print("Original dataset size: ", len(ds))

    # mean solution line length needs to be >= 3
    ds = ds.filter(lambda x: np.mean([len(s.strip().split("\n"))
                   for s in x["solutions"]]) >= args.min_solution_line_length, num_proc=os.cpu_count())
    print("After mean solution line length >= 3: ", len(ds))

    # need to have enough tests, at least 10
    ds = ds.filter(lambda x: len(json.loads(x["input_output"])[
                   "inputs"]) >= args.min_tests, num_proc=os.cpu_count())
    print("After having at least 10 tests: ", len(ds))

    # strip all solutions
    ds = ds.map(lambda x: {"solutions": [s.strip()
                for s in x["solutions"]]}, num_proc=os.cpu_count())

    # remove any solution that is not parsable or is too long
    ds = ds.map(lambda x: {"solutions": [
                s for s in x["solutions"] if does_parse(s) and len(tokenizer.encode(s)) <= args.max_tokens]}, num_proc=os.cpu_count())

    # filter to have at least 1 solution
    ds = ds.filter(lambda x: len(x["solutions"]) >= args.min_solutions)
    print("Has at least one solution: ", len(ds))

    ds = ds.map(patch_tests, num_proc=os.cpu_count())

    total_size = 0
    for ex in ds:
        total_size += len(ex["solutions"])

    print("Total number of solutions: ", total_size)

    # save the dataset
    ds.save_to_disk(args.output)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="taco_cleaned")
    parser.add_argument("--min_solution_line_length", type=int, default=3)
    parser.add_argument("--min_tests", type=int, default=10)
    parser.add_argument("--min_solutions", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()
    main(args)
