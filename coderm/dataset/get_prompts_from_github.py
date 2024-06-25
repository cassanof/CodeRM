import datasets


def get_prompt(code):
    # find first docstring
    d_start_idx = code.find('"""')
    d_end_idx = code.find('"""', d_start_idx + 3)
    if d_start_idx == -1 or d_end_idx == -1:
        raise ValueError("No docstring found")
    prompt = code[d_start_idx + 3:d_end_idx].strip()
    # unindent by 4 spaces
    prompt = prompt.replace("\n    ", "\n")
    prompt = '"""\n' + prompt + '\n"""'
    return prompt


def main(args):
    base_dataset = datasets.load_dataset(args.base_dataset, split="train")
    dataset = datasets.load_dataset(args.dataset, split="train")
    new_ds = []
    for ex in dataset.shuffle().select(range(int(len(base_dataset) * args.proportion))):
        p = get_prompt(ex["content"])
        new_ds.append({"prompt": p})

    for ex in base_dataset:
        new_ds.append({"prompt": ex["prompt"]})

    ds = datasets.Dataset.from_list(new_ds)
    print(ds)
    ds.push_to_hub(args.push, private=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="bigcode/python-stack-v1-functions-filtered-sc2")
    parser.add_argument("--base-dataset", type=str,
                        default="codegenning/taco-rl")
    parser.add_argument("--proportion", type=float, default=0.5)
    parser.add_argument("--push", type=str,
                        default="codegenning/taco-rl-mixed")
    args = parser.parse_args()
    main(args)
