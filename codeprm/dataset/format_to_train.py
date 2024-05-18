import datasets
from codeprm.model import py_prompt
import random


def main(args):
    ds = datasets.load_from_disk(args.dataset_path)
    print("Original ds:")
    print(ds)
    content = []
    solutions = []
    mark = args.mask_loss_mark if args.mask_loss_mark else ""
    for ex in ds:
        if args.strategy == "all":
            for sol in ex["solutions"]:
                content.append(py_prompt(ex["question"], mark + sol))
                solutions.append(sol)
        elif args.strategy == "random":
            sol = random.choice(ex["solutions"])
            content.append(py_prompt(ex["question"], mark + sol))
            solutions.append(sol)
        elif args.strategy == "high-loc":
            # picks randomly in the top 75% of solutions by LoC
            sols = ex["solutions"]
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

    ds = datasets.Dataset.from_dict(
        {"content": content, "solutions": solutions})
    print("New ds:")
    print(ds)
    print("Printing one example:")
    print(ds[0]["content"])
    ds.push_to_hub(args.push, private=True)
    print("IMPORTANT: Remember to MinHash-dedup the dataset before training! Only dedup based on 'solutions' column.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--push", type=str, required=True)
    parser.add_argument("--mask_loss_mark", type=str, default=None)
    parser.add_argument("--strategy", type=str, default="all",
                        # TODO: maybe add a strat that takes account of LoC distribution
                        choices=["all", "random", "high-loc"])
    args = parser.parse_args()
    random.seed(42)
    main(args)
