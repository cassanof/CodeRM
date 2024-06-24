import datasets
from coderm.model import py_prompt
import random
from pathlib import Path


def main(args):
    if Path(args.dataset_path).exists():
        ds = datasets.load_from_disk(args.dataset_path)
    else:
        ds = datasets.load_dataset(args.dataset_path, split="train")

    print("Original ds:")
    print(ds)
    content = []
    solutions = []
    mark = args.mask_loss_mark if args.mask_loss_mark else ""
    for ex in ds:
        sols = ex[args.sols_col]
        if args.strategy == "all" or args.strategy == "clip":
            if args.strategy == "clip":
                sols = sols[:args.clip_n]

            for sol in sols:
                content.append(py_prompt(ex["question"], mark + sol))
                solutions.append(sol)
        elif args.strategy == "random":
            sol = random.choice(sols)
            content.append(py_prompt(ex["question"], mark + sol))
            solutions.append(sol)
        elif args.strategy == "high-loc":
            # picks randomly in the top 75% of solutions by LoC
            locs = [len(sol.split("\n")) for sol in sols]
            locs_sorted = sorted(enumerate(locs), key=lambda x: x[1])
            n = len(locs)
            top_75 = locs_sorted[n // 4:]
            idx = random.choice(top_75)[0]
            sol = sols[idx]
            content.append(py_prompt(ex["question"], mark + sol))
            solutions.append(sol)
        else:
            raise ValueError("Invalid strategy: " + args.strategy)

    if args.rlxf:
        # split prompt from solution
        prompts = []
        for c in content:
            # find index of second '"""'
            first = c.find('"""')
            assert first != -1
            second = c.find('"""', first + 3)
            assert second != -1
            prompts.append(c[:second + 3])
        ds = datasets.Dataset.from_dict(
            {"prompt": prompts, "response": solutions})
        print("Printing one example:")
        print(ds[0]["prompt"])
        print(ds[0]["response"])
    else:
        ds = datasets.Dataset.from_dict(
            {"content": content, "solution": solutions})
        print("Printing one example:")
        print(ds[0]["content"])

    print("New ds:")
    print(ds)
    ds.push_to_hub(args.push, private=True)
    print("IMPORTANT: Remember to MinHash-dedup the dataset before training! Only dedup based on 'solution' (or 'response' for rlxf) column.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--push", type=str, required=True)
    parser.add_argument("--mask_loss_mark", type=str, default=None)
    parser.add_argument("--sols_col", type=str, default="solutions")
    parser.add_argument("--rlxf", action="store_true")
    parser.add_argument("--strategy", type=str, default="all",
                        choices=["all", "clip", "random", "high-loc"])
    parser.add_argument("--clip-n", type=int, default=500)
    args = parser.parse_args()
    random.seed(42)
    main(args)
