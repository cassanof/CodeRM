"""
Splits the given result file into a separate files for each difficulty level.
"""
import datasets
from codeprm.utils import gunzip_json_read, gunzip_json_write
from pathlib import Path
from tqdm import tqdm


def main(args):
    path = Path(args.input)
    obj = gunzip_json_read(path)
    assert obj is not None, f"Failed to read {path}"

    if args.taco_dataset is None:
        dataset_name = obj["dataset_name"]
    else:
        dataset_name = args.taco_dataset

    dataset = datasets.load_dataset(dataset_name, split="test")

    index_to_diff = {}
    diffs = set()
    for i, item in enumerate(dataset):
        diff = item["difficulty"]
        index_to_diff[i] = diff
        diffs.add(diff)

    difficulty_to_ds = {}
    for d in diffs:
        obj_no_items = {"items": []}
        for key, value in obj.items():
            if key != "items":
                obj_no_items[key] = value
        difficulty_to_ds[d] = obj_no_items

    for i, item in enumerate(obj["items"]):
        difficulty = index_to_diff[i]
        difficulty_to_ds[difficulty]["items"].append(item)

    for diff, ds in tqdm(difficulty_to_ds.items(), desc="Writing files"):
        stem = path.stem.split(".")[0]
        output_path = path.parent / f"{stem}_{diff}.json.gz"
        gunzip_json_write(output_path, ds)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        type=str,
        help="Input file"
    )
    parser.add_argument(
        "--taco-dataset",
        type=str,
        help="Dataset used for taco evaluation. If none, it will be derived from the input file",
        default=None,
    )
    args = parser.parse_args()
    main(args)
