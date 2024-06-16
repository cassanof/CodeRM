import datasets
from codeprm.prompts import py_prompt


def main(args):
    train_dataset = datasets.load_dataset(
        args.train, split="train")
    test_dataset = datasets.load_dataset(
        args.test, split="test")

    # shuffle to avoid bias
    train_dataset = train_dataset.shuffle()
    test_dataset = test_dataset.shuffle()

    train_fmt = []
    test_fmt = []

    for ex in train_dataset:
        post = ""
        if "starter_code" not in ex:
            post = "\n"
        p = py_prompt(ex["question"], ex["starter_code"]) + post
        train_fmt.append({"prompt": p})

    for ex in test_dataset:
        post = ""
        if "starter_code" not in ex:
            post = "\n"
        p = py_prompt(ex["question"], ex["starter_code"]) + post
        test_fmt.append({"prompt": p})

    ds = {
        "train": datasets.Dataset.from_list(train_fmt),
        "test": datasets.Dataset.from_list(test_fmt),
    }
    ds = datasets.DatasetDict(ds)
    ds.push_to_hub(args.push, private=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str,
                        default="cassanof/taco_cleaned_all")
    parser.add_argument("--test", type=str,
                        default="cassanof/livecodebench_lite_contaminated")
    parser.add_argument("--push", type=str, default="codegenning/taco-rl")
    args = parser.parse_args()
    main(args)
