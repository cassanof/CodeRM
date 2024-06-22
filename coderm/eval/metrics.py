"""
Takes in a result file and spits out the pass@k metrics.
"""
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import random
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


def approximate_perms(n, max_n, max_perms=100):
    # we average multiple random permutations for n > 1
    # we approximate the number of permutations required based on n and len(items[0]["results"])
    # the closer n is to len(items[0]["results"]), the less permutations are required
    max_perms = 100
    if n is None or n == max_n:
        perms = 1
    else:
        perms = int(max_perms * (1 - (n / max_n)) + 1)
        perms = min(perms, max_perms)

    return perms


def get_orm_acc(items, prod=None, n=None, k=1, perms=None) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculates the ORM accuracy, if the results contain ORM labels.

    The prod parameter allows to take the product of the logprobs provided by the generator model.
    Three cases are supported:
    - None: original ORM scores are used
    - "unnormalized": math.exp(cumulative_logprob) * orm_score is used
    - "normalized": math.exp(cumulative_logprob / num_tokens) * orm_score is used

    The n parameter allows to consider only the first N samples for ORM accuracy.
    """

    random.seed(42)
    if perms is None:
        perms = approximate_perms(n, len(items[0]["results"]))

    orm_acc = 0
    public_acc = 0
    for _ in range(perms):
        correct_pass = []
        correct_with_public_pass = []

        for item in items:
            results = item["results"]
            if n is not None:
                assert n > 0, "n parameter should be > 0"
                results = random.sample(results, n)

            score_results = []
            score_results_with_public = []
            for result in results:
                if "orm_1_score" not in result:
                    return None, None  # No labels found

                score = result["orm_1_score"]

                # reshape score
                if prod is not None:
                    score *= np.exp(result["cumulative_logprob"])
                elif prod == "normalized":
                    score *= np.exp(result["cumulative_logprob"] /
                                    result["num_tokens"])

                score_results.append((score, result["passing"]))
                if "passing_public" not in result:  # case combined with public execution labels
                    score_results_with_public = None
                elif result["passing_public"]:
                    assert score_results_with_public is not None
                    score_results_with_public.append(
                        (score, result["passing"]))

            score_results.sort(key=lambda x: x[0], reverse=True)
            top_k = score_results[:k]
            correct_pass.append(any(passing for _, passing in top_k))

            if score_results_with_public is not None:
                score_results_with_public.sort(
                    key=lambda x: x[0], reverse=True)
                top_k_with_public = score_results_with_public[:k]
                correct_with_public_pass.append(
                    any(passing for _, passing in top_k_with_public))

        orm_acc += np.mean(correct_pass)
        if correct_with_public_pass:
            public_acc += np.mean(correct_with_public_pass)
        else:
            public_acc = None

    return orm_acc / perms, public_acc / perms if public_acc is not None else None


def get_public_acc(items, n=None, k=1, perms=None) -> Optional[float]:
    """
    Calculates the public@n accuracy, if the results contain public labels.

    The consider parameter allows to consider only the first N samples for public accuracy.
    """
    random.seed(42)
    public_acc = 0

    if perms is None:
        perms = approximate_perms(n, len(items[0]["results"]))

    for _ in range(perms):
        correct = []
        for item in items:
            results = item["results"]
            if n is not None:
                assert n > 0, "public consider parameter should be > 0"
                results = results[:n]

            for result in results:
                if "passing_public" not in result:
                    return None  # No public labels found

                passing_public = []
                not_passing_public = []
                for result in results:
                    if result["passing_public"]:
                        passing_public.append(result)
                    else:
                        not_passing_public.append(result)

                res = passing_public + not_passing_public
                top_k = res[:k]
                correct.append(any(result["passing"] for result in top_k))

        public_acc += np.mean(correct)

    return public_acc / perms


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

    orm_acc, orm_acc_public = get_orm_acc(items, prod=orm_prod, n=orm_n, k=k)
    orm_acc = round(orm_acc * 100, 4) if orm_acc is not None else "N/A"
    orm_acc_public = round(orm_acc_public * 100,
                           4) if orm_acc_public is not None else "N/A"

    public_acc = get_public_acc(items, n=public_n, k=k)
    public_acc = round(
        public_acc * 100, 4) if public_acc is not None else "N/A"

    return f"{file.stem},{size},{len(items[0]['results'])},{k},{mean_pass_k},{orm_acc},{public_acc},{orm_acc_public}"


def main(args):
    header = "name,dataset size,n,k,avg pass@k,orm@{k|n},public@{k|n},orm+public@{k|n}"
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
    args = parser.parse_args()
    main(args)
