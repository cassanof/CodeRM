import datasets
from codeprm.model import py_prompt


def main(args):
    ds = datasets.load_from_disk(args.dataset_path)
    print("Original ds:")
    print(ds)
    content = []
    solutions = []
    mark = args.mask_loss_mark if args.mask_loss_mark else ""
    for ex in ds:
        for sol in ex["solutions"]:
            content.append(py_prompt(ex["question"], mark + sol))
            solutions.append(sol)

    ds = datasets.Dataset.from_dict({"content": content, "solutions": solutions})
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
    args = parser.parse_args()
    main(args)
