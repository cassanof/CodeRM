"""
Takes in a result file and spits out the pass@k metrics.
"""
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from coderm.utils import gunzip_json_read


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


def get_orm_acc(items, prod=None, n=None, ensemble="min") -> Tuple[Optional[float], Optional[float]]:
    """
    Calculates the ORM accuracy, if the results contain ORM labels.

    The prod parameter allows to take the product of the logprobs provided by the generator model.
    Three cases are supported:
    - None: original ORM scores are used
    - "unnormalized": math.exp(cumulative_logprob) * orm_score is used
    - "normalized": math.exp(cumulative_logprob / num_tokens) * orm_score is used

    The consider parameter allows to consider only the first N samples for ORM accuracy.

    The ensemble parameter allows to combine multiple ORM scores in a single result.
    """
    def shape_score(score, result):
        if prod == "unnormalized":
            score *= np.exp(result["cumulative_logprob"])
        elif prod == "normalized":
            score *= np.exp(result["cumulative_logprob"] /
                            result["num_tokens"])
        return score

    correct = 0
    correct_with_public = 0
    total = 0

    for item in items:
        max_score = -1
        max_res = None

        max_score_with_public = -1
        max_res_with_public = None

        results = item["results"]
        if n is not None:
            assert n > 0, "orm consider parameter should be > 0"
            results = results[:n]

        for result in results:
            if "orm_1_score" in result:  # single ORM
                score = result["orm_1_score"]
            elif "orms" in result:  # multiple ORMs
                scores = []
                for r in result["orms"]:
                    scores.append(r["orm_1_score"])
                if ensemble == "min":
                    score = min(scores)
                elif ensemble == "mean":
                    score = np.mean(scores)
                else:
                    raise ValueError(f"Unknown ensemble method {ensemble}")
            else:
                return None, None  # No labels found

            score = shape_score(score, result)
            if score > max_score:
                max_score = score
                max_res = result

            if "passing_public" not in result:
                correct_with_public = None
            elif result["passing_public"]:
                if score > max_score_with_public:
                    max_score_with_public = score
                    max_res_with_public = result

        assert max_res is not None, "No ORM labels found"

        if max_res["passing"]:
            correct += 1

        if max_res_with_public and max_res_with_public["passing"]:
            assert correct_with_public is not None
            correct_with_public += 1

        total += 1

    return correct / total, ((correct_with_public / total) if correct_with_public is not None else None)


def get_public_acc(items, n=None) -> Optional[float]:
    """
    Calculates the public@n accuracy, if the results contain public labels.

    The consider parameter allows to consider only the first N samples for public accuracy.
    """
    correct = 0
    total = 0
    for item in items:
        passing_public = None

        results = item["results"]
        if n is not None:
            assert n > 0, "public consider parameter should be > 0"
            results = results[:n]

        for result in results:
            if "passing_public" not in result:
                return None  # No public labels found

            if result["passing_public"]:
                passing_public = result

        if passing_public and passing_public["passing"]:
            correct += 1

        total += 1

    return correct / total


def per_file_metrics(file: Path, k: int, orm_prod=None, orm_n=None, public_n=None) -> str:
    if file.is_dir():
        import datasets
        ds = datasets.load_from_disk(file)
        items = ds.to_list()
    else:
        obj = gunzip_json_read(file)
        assert obj is not None, f"Failed to read {file}"
        items = obj["items"]

    size = len(items)

    pass_ks = get_pass_ks(items, k)
    mean_pass_k = round(np.mean(pass_ks) * 100, 4)

    orm_acc, orm_acc_public = get_orm_acc(items, prod=orm_prod, n=orm_n)
    orm_acc = round(orm_acc * 100, 4) if orm_acc is not None else "N/A"
    orm_acc_public = round(orm_acc_public * 100,
                           4) if orm_acc_public is not None else "N/A"

    public_acc = get_public_acc(items, n=public_n)
    public_acc = round(
        public_acc * 100, 4) if public_acc is not None else "N/A"

    return f"{file.stem},{size},{len(items[0]['results'])},{k},{mean_pass_k},{orm_acc},{public_acc},{orm_acc_public}"


def main(args):
    header = "name,dataset size,n,k,avg pass@k,orm@{1|n},public@{1|n},orm+public@{1|n}"
    print(header)
    for file in args.inputs:
        print(
            per_file_metrics(
                file=Path(file),
                k=args.k,
                orm_prod=args.orm_prod,
                orm_n=args.orm_n,
                public_n=args.public_n
            )
        )


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
        "--orm-n",
        type=int,
        default=None,
        help="How many samples should be considered for ORM accuracy",
    )
    parser.add_argument(
        "--public-n",
        type=int,
        default=None,
        help="How many samples should be considered for public@n accuracy",
    )
    parser.add_argument(
        "--orm-ensemble",
        type=str,
        choices=["min", "mean"],
    )
    args = parser.parse_args()
    main(args)
