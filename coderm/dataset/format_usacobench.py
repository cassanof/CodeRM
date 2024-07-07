from typing import Tuple
import json
from pathlib import Path
import datasets
import re
from tqdm import tqdm


def main(args):
    raw_dataset = datasets.load_from_disk(args.dataset)
    base_testdir = Path(args.dataset) / "tests"
    # rename "problem_level" to "difficulty"
    new_ds = []
    for ex in tqdm(raw_dataset, total=len(raw_dataset)):
        eid = ex["problem_id"]
        difficulty = ex["problem_level"]
        testsdir = base_testdir / eid
        num_tests = ex["num_tests"]
        tests = {"inputs": [], "outputs": []}
        files = [f for f in testsdir.glob("*")]
        for i in range(1, num_tests+1):
            # find the two files (input and output) with the digit i in the filename
            id_files = [f for f in files if f.name.startswith(
                f"{i}.") or f.name.endswith(f".{i}")]
            assert len(id_files) == 2, f"Expected 2 files for test {
                i}, got {id_files} - {files}"
            # output has "out", input has "in"
            output_file = [f for f in id_files if "o" in f.name.lower()]
            input_file = [f for f in id_files if "i" in f.name.lower()]
            assert len(output_file) == 1, f"Expected 1 output file for test {i}, got {output_file} - {files}"
            assert len(input_file) == 1, f"Expected 1 input file for test {i}, got {input_file} - {files}"
            tests["inputs"].append(input_file[0].read_text())
            tests["outputs"].append(output_file[0].read_text())
        jsonified = json.dumps(tests)
        new_ds.append({"id": eid, "difficulty": difficulty, "input_output": jsonified, "question": ex["description"]})
    new_ds = datasets.Dataset.from_list(new_ds)
    print(new_ds)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="./on_disk/usaco_v3")
    args = parser.parse_args()
    main(args)
