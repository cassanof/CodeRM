"""
Takes in a result file and spits out the pass@k metrics.
"""

from pathlib import Path
from typing import Optional
import numpy as np
from codeprm.utils import gunzip_json_read


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Calculates 1 - comb(n - c, k) / comb(n, k).
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def get_pass_ks(items, k):
    n_comps = len(items[0]["results"])
    assert n_comps >= k, f"Completion limit {n_comps} is less than k {k}"
    assert all(len(item["results"]) ==
               n_comps for item in items), "All items should have the same number of completions"
    pass_ks = []
    for item in items:
        correct = 0
        for result in item["results"]:
            if result["passing"]:
                correct += 1

        score = pass_at_k(n_comps, correct, k)
        pass_ks.append(score)
    return pass_ks


def get_orm_acc(items) -> Optional[float]:
    """
    Calculates the ORM accuracy, if the results contain ORM labels.
    """
    correct = 0
    total = 0
    for item in items:
        max_score = -1
        max_res = None

        for result in item["results"]:
            if "orm_score" not in result:
                return None  # ORM score not found

            if result["orm_label"] == 1 and result["orm_score"] > max_score:
                max_score = result["orm_score"]
                max_res = result

        if max_res is None:
            max_res = item["results"][0]  # first one will do

        if max_res["orm_label"] == 1 and max_res["passing"]:
            correct += 1
        total += 1

    return round(correct / total * 100, 4)


def per_file_metrics(file: Path, k: int) -> str:
    obj = gunzip_json_read(file)
    assert obj is not None, f"Failed to read {file}"

    items = obj["items"]
    size = len(items)

    pass_ks = get_pass_ks(items, k)
    orm_acc = get_orm_acc(items)

    return f"{file.stem},{size},{len(items[0]['results'])},{k},{np.mean(pass_ks)},{orm_acc}"


def main(args):
    header = "name,dataset size,num completions,k,avg pass@k,orm acc"
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
