import datasets
import numpy as np
import os
import json
import sys
import ast
sys.set_int_max_str_digits(0)

def does_parse(code):
    try:
        ast.parse(code)
        return True
    except:
        return False

ds = datasets.load_dataset("BAAI/TACO", split="train", trust_remote_code=True)
# json load the solutions
ds = ds.map(lambda x: {"solutions": json.loads(
    x["solutions"])}, num_proc=os.cpu_count())
print("Original dataset size: ", len(ds))

# mean solution line length needs to be >= 3
ds = ds.filter(lambda x: np.mean([len(s.strip().split("\n")) for s in x["solutions"]]) >= 3, num_proc=os.cpu_count())
print("After mean solution line length >= 3: ", len(ds))


# need to have enough tests, at least 10
ds = ds.filter(lambda x: len(json.loads(x["input_output"])["inputs"]) >= 10, num_proc=os.cpu_count())
print("After having at least 10 tests: ", len(ds))

# remove any solution that is not parsable
ds = ds.map(lambda x: {"solutions": [s for s in x["solutions"] if does_parse(s)]}, num_proc=os.cpu_count())
# filter to have at least 1 solution
ds = ds.filter(lambda x: len(x["solutions"]) > 0)
print("Has at least one solution: ", len(ds))

# patch up solutions with fn_name to have single output, idk why output is a list
def patch(ex):
    tests = json.loads(ex["input_output"])
    if "fn_name" not in tests:
        return ex
    
    outputs = []
    for o in tests["outputs"]:
        outputs.append(o[0])
    tests["outputs"] = outputs
    return {"input_output": json.dumps(tests)}

ds = ds.map(patch, num_proc=os.cpu_count())

total_size = 0
for ex in ds:
    total_size += len(ex["solutions"])

print("Total number of solutions: ", total_size)

# save the dataset
ds.save_to_disk("taco_cleaned")
