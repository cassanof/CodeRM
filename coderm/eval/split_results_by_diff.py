"""
Splits the given result file into a separate files for each difficulty level.
"""
from coderm.utils import gunzip_json_read, gunzip_json_write
from pathlib import Path
from tqdm import tqdm


def main(args):
    path = Path(args.input)
    obj = gunzip_json_read(path)
    assert obj is not None, f"Failed to read {path}"

    diffs = set()
    for item in obj["items"]:
        diffs.add(item["difficulty"])

    difficulty_to_ds = {}
    for d in diffs:
        obj_no_items = {"items": []}
        for key, value in obj.items():
            if key != "items":
                obj_no_items[key] = value
        difficulty_to_ds[d] = obj_no_items

    for item in obj["items"]:
        difficulty_to_ds[item["difficulty"]]["items"].append(item)

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
    args = parser.parse_args()
    main(args)
