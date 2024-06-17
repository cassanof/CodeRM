import datasets
from transformers import AutoTokenizer
from codeprm.prompts import py_prompt

ORDER = ["EASY", "MEDIUM", "MEDIUM_HARD", "HARD", "VERY_HARD"]


def main(args):
    train_dataset = datasets.load_dataset(
        "BAAI/TACO", split="train+test", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # dedup by question
    print("Deduping by question. Before: ", len(train_dataset))
    new_train = []
    qset = set()
    for ex in train_dataset:
        q = ex["question"]
        if q not in qset:
            new_train.append(ex)
            qset.add(q)
    train_dataset = datasets.Dataset.from_list(new_train)
    print("After dedup: ", len(train_dataset))

    # filter out long examples
    train_dataset = train_dataset.filter(
        lambda x: len(tokenizer.encode(x["question"])) < args.max_tokens)
    print("After filtering for max length: ", len(train_dataset))

    test_dataset = datasets.load_dataset(
        args.test, split="test")

    if args.order == "curriculum":
        new_train = []
        print("Curriculum order")
        for diff in ORDER:
            fil = train_dataset.filter(
                lambda x: x["difficulty"] == diff).to_list()
            print(f"{diff}: {len(fil)}")
            new_train.extend(fil)

        # evenly distribute UNKNOWN_DIFFICULTY
        unknown = train_dataset.filter(
            lambda x: x["difficulty"] == "UNKNOWN_DIFFICULTY").to_list()
        print("Adding unknown difficulty: ", len(unknown))
        for i, ex in enumerate(unknown):
            new_train.insert(i * (len(new_train) // len(unknown)), ex)

        train_dataset = datasets.Dataset.from_list(new_train)

    elif args.order == "random":
        # shuffle to avoid bias
        train_dataset = train_dataset.shuffle()
    else:
        raise ValueError(f"Unknown order: {args.order}")

    # just shuffle test set
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
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--tokenizer", type=str,
                        default="bigcode/starcoder2-15b")
    parser.add_argument("--push", type=str, default="codegenning/taco-rl")
    parser.add_argument("--order", type=str, choices=["random", "curriculum"],
                        default="random")
    args = parser.parse_args()
    main(args)
