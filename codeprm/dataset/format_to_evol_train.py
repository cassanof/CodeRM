import datasets
from codeprm.prompts import py_prompt_evolve, py_prompt
import random


def main(args):
    ds = datasets.load_from_disk(args.dataset_path)
    print("Original ds:")
    print(ds)
    contents = []
    questions = []
    befores = []
    afters = []
    mark = args.mask_loss_mark if args.mask_loss_mark else ""
    under_2 = 0
    for ex in ds:
        sols = ex[args.sols_col]

        if len(sols) < 2:
            under_2 += 1
            continue

        q = ex["question"]
        for i in range(1, len(sols)):
            before = sols[i-1]
            after = sols[i]
            questions.append(q)
            befores.append(before)
            afters.append(after)

            base_prompt = py_prompt(q, mark + before)
            content = py_prompt_evolve(base_prompt, after)
            contents.append(content)

    print(f"Total examples: {len(ds)}")
    print(f"Problems lost due to under 2 solutions: {under_2}")
    ds = datasets.Dataset.from_dict(
        {"content": contents, "question": questions, "before": befores, "after": afters})
    print("Printing one example:")
    print(ds[0]["content"])
    print("New ds:")
    print(ds)
    ds.push_to_hub(args.push, private=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--push", type=str, required=True)
    parser.add_argument("--mask_loss_mark", type=str, default=None)
    parser.add_argument("--sols_col", type=str, default="solutions")
    args = parser.parse_args()
    random.seed(42)
    main(args)
