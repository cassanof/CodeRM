from typing import Any, Optional, Tuple
from coderm.prompts import py_prompt
import random
import datasets
from tqdm import tqdm


def exec_selection(results) -> Tuple[Optional[str], Optional[str], Optional[Any], Optional[Any]]:
    chosen = None
    rejected = None
    rejected_error = None
    for r in results:
        code = r["code"]
        if r["passing"]:
            chosen = code
        else:
            rejected = code
            rejected_error = r["output"]
    return chosen, rejected, None, rejected_error


def best_worst_score_selection(results) -> Tuple[Optional[str], Optional[str], Optional[Any], Optional[Any]]:
    chosen = None
    rejected = None
    chosen_score = -1
    rejected_score = 1
    for r in results:
        code = r["code"]
        score = r["orm_1_score"]
        if score >= chosen_score:
            chosen = code
            chosen_score = score
        if score <= rejected_score:
            rejected = code
            rejected_score = score
    assert chosen != rejected, "Chosen and rejected are the same"
    return chosen, rejected, chosen_score, rejected_score


def best_random_score_selection(results) -> Tuple[Optional[str], Optional[str], Optional[Any], Optional[Any]]:
    chosen = None
    chosen_score = -1
    for r in results:
        code = r["code"]
        score = r["orm_1_score"]
        if score >= chosen_score:
            chosen = code
            chosen_score = score
    rej_obj = random.choice([r for r in results if r["code"] != chosen])
    rejected = rej_obj["code"]
    rejected_score = rej_obj["orm_1_score"]
    assert chosen != rejected, "Chosen and rejected are the same"
    return chosen, rejected, chosen_score, rejected_score


def main(args):
    random.seed(42)
    ds = datasets.load_from_disk(args.input)

    prompt_to_nat = {}
    if args.natural is not None:
        natural_ds = datasets.load_from_disk(args.natural)
        for ex in natural_ds:
            if len(ex[args.natural_col]) > 0:
                prompt_to_nat[ex["question"]] = ex[args.natural_col][0]

    new_ds = []
    used_nat = 0
    used_synth = 0
    for ex in tqdm(ds, total=len(ds)):
        results = ex["results"][:args.n]
        if args.selection == "exec":
            chosen, rejected, chosen_info, rejected_info = exec_selection(
                results)
        elif args.selection == "best-worst-score":
            chosen, rejected, chosen_info, rejected_info = best_worst_score_selection(
                results)
        elif args.selection == "best-random-score":
            chosen, rejected, chosen_info, rejected_info = best_random_score_selection(
                results)
        else:
            raise ValueError(f"Unknown selection method: {args.selection}")

        if chosen is None:
            # attempt to get a natural solution
            chosen = prompt_to_nat.get(ex["prompt"], None)
            if chosen is not None:
                used_nat += 1
        else:
            used_synth += 1

        if chosen is None or rejected is None:
            continue

        starter = ex["starter_code"]
        prompt = py_prompt(ex["prompt"], starter)
        if starter is None or starter == "":
            prompt += "\n"

        defs = {
            "prompt": prompt,
            "text_chosen": chosen,
            "text_rejected": rejected,
            "chosen_info": chosen_info,
            "rejected_info": rejected_info,
        }
        new_ds.append(defs)

    # dedup by prompt
    new_ds = list({v["prompt"]: v for v in new_ds}.values())
    print(f"Final dataset size: {len(new_ds)}")
    print(f"Used natural: {used_nat}")
    print(f"Used synthetic: {used_synth}")
    final_ds = datasets.Dataset.from_list(new_ds)
    final_ds.push_to_hub(args.push, private=True, split=args.push_split)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="Input dataset path for the results")
    parser.add_argument("--natural", type=str, required=False,
                        help="Input dataset with natural chosens")
    parser.add_argument("--natural_col", type=str, default="reasoning_steps")
    parser.add_argument("--selection", type=str,
                        choices=["exec", "best-worst-score", "best-random-score"], default="exec")
    parser.add_argument("-n", type=int, default=99999,
                        help="Max number of samples per example to use")
    parser.add_argument("--push", type=str, required=True,
                        help="Push dataset path for the ORM")
    parser.add_argument("--push-split", type=str, default="train",
                        help="The split for the pushed dataset")
    args = parser.parse_args()
    main(args)
