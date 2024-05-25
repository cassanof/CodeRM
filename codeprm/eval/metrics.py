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


def get_orm_acc(items, prod=None, consider=None) -> Optional[float]:
    """
    Calculates the ORM accuracy, if the results contain ORM labels.

    The prod parameter allows to take the product of the logprobs provided by the generator model.
    Three cases are supported:
    - None: original ORM scores are used
    - "unnormalized": math.exp(cumulative_logprob) * orm_score is used
    - "normalized": math.exp(cumulative_logprob / num_tokens) * orm_score is used

    The consider parameter allows to consider only the first N samples for ORM accuracy.
    """
    correct = 0
    total = 0
    for item in items:
        max_score = -1
        max_res = None

        results = item["results"]
        if consider is not None:
            assert consider > 0, "orm consider parameter should be > 0"
            results = results[:consider]

        for result in results:
            if "orm_score" not in result:
                return None  # ORM score not found

            if result["orm_label"] == 1 and result["orm_score"] > max_score:
                score = result["orm_score"]
                if prod == "unnormalized":
                    score *= np.exp(result["cumulative_logprob"])
                elif prod == "normalized":
                    score *= np.exp(result["cumulative_logprob"] /
                                    result["num_tokens"])
                max_score = score
                max_res = result

        if max_res is None:
            max_res = item["results"][0]  # first one will do

        if max_res["orm_label"] == 1 and max_res["passing"]:
            correct += 1
        total += 1

    return correct / total


def per_file_metrics(file: Path, k: int, orm_prod=None, orm_consider=None) -> str:
    obj = gunzip_json_read(file)
    assert obj is not None, f"Failed to read {file}"

    items = obj["items"]
    size = len(items)

    pass_ks = get_pass_ks(items, k)
    mean_pass_k = round(np.mean(pass_ks) * 100, 4)
    orm_acc = get_orm_acc(items, prod=orm_prod, consider=orm_consider)
    orm_acc = round(orm_acc * 100, 4) if orm_acc is not None else "N/A"

    return f"{file.stem},{size},{len(items[0]['results'])},{k},{mean_pass_k},{orm_acc}"


def main(args):
    header = "name,dataset size,num completions,k,avg pass@k,orm acc"
    print(header)
    for file in args.inputs:
        print(per_file_metrics(Path(file), args.k,
              orm_prod=args.orm_prod, orm_consider=args.orm_consider))


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
    parser.add_argument(
        "--orm-prod",
        type=str,
        default=None,
        choices=[None, "unnormalized", "normalized"],
        help="Product of logprobs for ORM accuracy"
    )
    parser.add_argument(
        "--orm-consider",
        type=int,
        default=None,
        help="How many samples should be considered for ORM accuracy",
    )
    args = parser.parse_args()
    main(args)
