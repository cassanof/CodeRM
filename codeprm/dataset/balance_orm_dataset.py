import datasets
import random


def main(args):
    ds = datasets.load_dataset(args.dataset, split=args.split)
    # balance the dataset
    passing_examples = [ex for ex in ds if ex['score'] == 1]
    failing_examples = [ex for ex in ds if ex['score'] == 0]
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
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--push", type=str, required=True)
    parser.add_argument("--push-split", type=str, default="train")
    args = parser.parse_args()
    main(args)
