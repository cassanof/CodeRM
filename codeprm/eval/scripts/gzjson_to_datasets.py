from codeprm.utils import gunzip_json_read
from pathlib import Path
import datasets


def main(args):
    data = gunzip_json_read(Path(args.input))
    assert data is not None, f"Failed to read {args.input}"
    items = data["items"]
    dataset = datasets.Dataset.from_list(items)
    dataset.save_to_disk(args.output)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="Input file path for .json.gz file")
    parser.add_argument("--output", type=str, required=True,
                        help="Output file path for the dataset")
    args = parser.parse_args()
    main(args)
