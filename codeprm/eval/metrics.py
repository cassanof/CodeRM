from pathlib import Path
import numpy as np
from codeprm.utils import gunzip_json_read


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Calculates 1 - comb(n - c, k) / comb(n, k).
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def per_file_metrics(file: Path, k: int) -> str:
    obj = gunzip_json_read(file)
    assert obj is not None, f"Failed to read {file}"
    
    items = obj["items"]
    n = obj["completion_limit"]
    assert n >= k, f"Completion limit {n} is less than k {k}"

    pass_ks = []
    for item in items:
        correct = 0
        for result in item["results"]:
            if result["passing"]:
                correct += 1

        pass_ks.append(pass_at_k(n, correct, k))

    return f"{file.stem},{n},{k},{np.mean(pass_ks)},{np.std(pass_ks)}"








def main(args):
    header = "name,n,k,avg pass@k,std pass@k"
    print(header)
    for file in args.inputs:
        print(per_file_metrics(Path(file), args.k))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inputs",
        type=str,
        nargs="+",
        help="Input files"
    )
    parser.add_argument(
        "-k",
        type=int,
        default=1,
        help="k for pass@k"
    )
    args = parser.parse_args()
    main(args)
