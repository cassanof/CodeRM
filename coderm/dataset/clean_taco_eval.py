import datasets
from coderm.dataset.clean_taco import patch_tests
import numpy as np
import os
import json


def main(args):
    ds = datasets.load_dataset(
        "BAAI/TACO", split=args.split, trust_remote_code=True)
    ds = ds.map(patch_tests, num_proc=os.cpu_count())
    # save the dataset
    ds.save_to_disk(args.output)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="taco_cleaned_eval")
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()
    main(args)
