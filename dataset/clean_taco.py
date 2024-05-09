import datasets
import os
import json
import sys
sys.set_int_max_str_digits(0)

IO_INSTRUMENTATION = """

"""

ds = datasets.load_dataset("BAAI/TACO", split="train", trust_remote_code=True)
ds = ds.map(lambda x: {"solutions": json.loads(
    x["solutions"])}, num_proc=os.cpu_count())
print("Original dataset size: ", len(ds))
# filter to have at least 1 solution
ds = ds.filter(lambda x: len(x["solutions"]) > 0)
print("Has at least one solution: ", len(ds))
# filter codewars, very low quality
ds = ds.filter(lambda x: x["source"] != "codewars", num_proc=os.cpu_count())
print("Filtered codewars: ", len(ds))

total_size = 0
for ex in ds:
    total_size += len(ex["solutions"])

print("Total number of solutions: ", total_size)


