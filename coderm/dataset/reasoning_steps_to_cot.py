from typing import List
from pathlib import Path
import datasets


def punctuation_join(lst: List[str]) -> str:
    """
    Joins a list of strings with ", " unless it's the last element or if there is already a punctuation mark.
    """
    result = ""
    for i, s in enumerate(lst):
        if i != 0:
            result += ", "
        if s[-1] not in [".", "!", "?"]:
            result += s
        else:
            result += s[:-1]
    return result


def cottify_reasoning_steps(code: str) -> List[str]:
    lines = code.split('\n')
    comments = []
    current_comment = []

    for line in lines:
        stripped_line = line.lstrip()
        if stripped_line.startswith('#'):
            comment_content = stripped_line[1:].strip()
            spaces = len(line) - len(stripped_line)
            if len(current_comment) == 0:
                current_comment.append(f"# {' ' * spaces}- {comment_content}")
            else:
                current_comment.append(comment_content)
        else:
            if current_comment:
                comments.append(punctuation_join(current_comment))
                current_comment = []

    if current_comment:
        comments.append(punctuation_join(current_comment))

    return comments


def reasoning_steps_to_cot(code) -> str:
    """"
    Takes a program with reasoning steps and converts it to a program with
    comments on top in a bullet point format.

    Example:
    Input:
    ```
    # This function adds 1 to the input
    def f(x):
        # We add 1 to the input
        return x + 1
    ```
    Output:
    ```
    # - This function adds 1 to the input
    #   - We add 1 to the input
    def f(x):
        return x + 1
    ```
    """
    lines = code.split("\n")
    cots = cottify_reasoning_steps(code)
    new_lines = []
    for line in lines:
        if not line.lstrip().startswith("#"):
            new_lines.append(line)

    cots = "\n".join(cots)
    new_lines = "\n".join(new_lines)
    return f"{cots}\n{new_lines}"


def main(args):
    if Path(args.dataset).exists():
        dataset = datasets.load_from_disk(args.dataset)
    else:
        dataset = datasets.load_dataset(args.dataset, split="train")

    if args.dataset_kind == "sft":
        dataset = dataset.map(lambda x: {
            "reasoning_steps": [reasoning_steps_to_cot(step) for step in x["reasoning_steps"]]
        })
    else:
        raise NotImplementedError(
            f"Dataset kind {args.dataset_kind} not implemented.")

    if args.push:
        dataset.push_to_hub(args.output, private=True)
    else:
        dataset.save_to_disk(args.output)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--dataset-kind", type=str,
                        default="sft", choices=["sft", "orm"])
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    main(args)
