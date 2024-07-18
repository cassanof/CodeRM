from codeprm.prompts import py_prompt
from codeprm.utils import strip_python_comments
import random
import datasets


def main(args):
    random.seed(42)
    ds = datasets.load_from_disk(args.input)

    new_ds = []
    total_passing = 0
    total_failing = 0
    for ex in ds:
        passing = []
        failing = []
        failing_reasons = []

        for r in ex["results"]:
            code = r["code"]
            if r["passing"]:
                passing.append(code)
            else:
                failing.append(code)
                failing_reasons.append(r["output"])

        total_passing += len(passing)
        total_failing += len(failing)

        to_sample = min(len(passing), len(failing), args.max_per_class)
        # sample one if there is only one
        to_sample_pass = max(to_sample, 1 if len(passing) > 1 else 0)
        to_sample_fail = max(to_sample, 1 if len(failing) > 1 else 0)

        def code_to_content(code):
            code = ex["starter_code"] + code
            if args.strip_comments:
                code = strip_python_comments(code)
            return py_prompt(ex["prompt"], code)

        defs = {
            "question": ex["prompt"],
            "starter_code": ex["starter_code"],
        }

        for code in passing[:to_sample_pass]:
            new_ds.append({"content": code_to_content(
                code), "score": 1, "solution": code, "output": None, **defs})
        for code, o in zip(failing[:to_sample_fail], failing_reasons[:to_sample_fail]):
            new_ds.append({"content": code_to_content(
                code), "score": 0, "solution": code, "output": o, **defs})

    print(f"Total passing examples before partial sampling: {total_passing}")
    print(f"Total failing examples before partial sampling: {total_failing}")

    # print stats
    print(f"Total examples: {len(ds)}")
    print(f"Total examples in new dataset: {len(new_ds)}")
    print(f"Passing examples: {sum(ex['score'] == 1 for ex in new_ds)}")
    print(f"Failing examples: {sum(ex['score'] == 0 for ex in new_ds)}")
    print()

    # balance the dataset
    passing_examples = [ex for ex in new_ds if ex['score'] == 1]
    failing_examples = [ex for ex in new_ds if ex['score'] == 0]
    min_examples = min(len(passing_examples), len(failing_examples))

    balanced_ds = random.sample(
        passing_examples, min_examples) + random.sample(failing_examples, min_examples)
    random.shuffle(balanced_ds)
    print(f"Total examples in balanced dataset: {len(balanced_ds)}")
    print("Passing examples in balanced dataset: " +
          f"{sum(ex['score'] == 1 for ex in balanced_ds)}")
    print("Failing examples in balanced dataset: " +
          f"{sum(ex['score'] == 0 for ex in balanced_ds)}")

    final_ds = datasets.Dataset.from_list(balanced_ds)
    final_ds.push_to_hub(args.push, private=True, split=args.push_split)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="Input dataset path for the results")
    parser.add_argument("--push", type=str, required=True,
                        help="Push dataset path for the ORM")
    parser.add_argument("--push-split", type=str, default="train",
                        help="The split for the pushed dataset")
    parser.add_argument("--max-per-class", type=int, default=50,
                        help="Max number of examples per class")
    parser.add_argument("--strip-comments", action="store_true",
                        help="Strip comments from the code")
    args = parser.parse_args()
    main(args)
