from typing import Optional, List, Tuple
import difflib
import json
from pyarrow.ipc import pa
from codeprm.mutation import mutate
import os
from codeprm.execution import smart_exec_tests, parse_time_limit
from codeprm.prompts import py_prompt
import datasets
import random


def get_reasoning_steps_indexed(code: str) -> List[Tuple[str, Tuple[int, int]]]:
    lines = code.split('\n')
    comments = []
    current_comment = []
    current_range_start = None
    last_code_line = -1  # track the last line of code before a new comment

    for index, line in enumerate(lines):
        stripped_line = line.lstrip()
        if stripped_line.startswith('#'):
            if current_comment and current_range_start is not None:
                # append the current comment and the range from start to the last code line before this comment
                comments.append((' '.join(current_comment),
                                (current_range_start, last_code_line)))
                current_comment = []
                current_range_start = None
            # append line to the current comment block
            comment_content = stripped_line[1:].strip()
            current_comment.append(comment_content)
        else:
            if current_comment and current_range_start is None:
                # start new code range after current comment block
                current_range_start = index
            last_code_line = index

    if current_comment and current_range_start is not None:
        # append the last comment block and range till the end of the file or the last code line
        comments.append((' '.join(current_comment),
                        (current_range_start, last_code_line)))
    elif current_comment:
        comments.append((' '.join(current_comment), (len(lines), len(lines))))

    return comments


def mutate_until_fail(code, tests, timeout, max_tries=10, executor="http://127.0.0.1:8000") -> Optional[str]:
    """
    Returns None in case no mutation is found that makes the code fail.
    """
    for _ in range(max_tries):
        mutated_code = mutate(code)
        if mutated_code == code:
            continue

        results = smart_exec_tests(
            mutated_code, tests, executor="http://127.0.0.1:8000", timeout=timeout)
        if not results[0]:
            return mutated_code

    return None


def get_mutated_step(code, mutated) -> Tuple[int, int]:
    # get line number of the mutation
    steps = get_reasoning_steps_indexed(code)
    ranges = [step[1] for step in steps]
    codesplit = code.splitlines()
    mutatedsplit = mutated.splitlines()
    diff = difflib.ndiff(codesplit, mutatedsplit)
    diff = list(diff)
    line_number = 0
    for line in diff:
        if line.startswith("+") or line.startswith("-") or line.startswith("?"):
            break
        line_number += 1
    # get range
    for r in ranges:
        if r[0] <= line_number <= r[1]+1:
            return r
    mutdiff = "\n".join(diff)
    raise ValueError(f"No range found for line number: {line_number}\n" +
                     f"Ranges: {ranges}\nMutation:\n{mutdiff}")


def mutate_example(ex, args):
    tests = json.loads(ex["input_output"])
    timeout = parse_time_limit(ex["time_limit"])

    mutated = []
    mutated_steps = []

    for code in ex[args.code_col]:
        mutated_code = mutate_until_fail(
            code, tests, executor=args.executor, timeout=timeout)
        if mutated_code:
            mutated.append(mutated_code)
            mutated_steps.append(get_mutated_step(code, mutated_code))

    return {"mutated": mutated, "mutation_step": mutated_steps}


def main(args):
    ds = datasets.load_dataset(args.dataset, split="train")
    if args.sample:
        ds = ds.select(range(args.sample))

    print(ds)
    total_steps = 0
    for example in ds:
        total_steps += len(example[args.code_col])

    print("Total steps:", total_steps)
    mutated_ds = ds.map(
        lambda ex: mutate_example(ex, args),
        num_proc=args.workers,
    )
    print(mutated_ds)
    total_mutated_steps = 0
    for example in mutated_ds:
        total_mutated_steps += len(example["mutated"])
    print("Total mutated steps:", total_mutated_steps)

    # format to ORM dataset: ['content', 'score', 'solution', 'question', 'starter_code', 'mutation_step']
    new_ds = []

    for ex in mutated_ds:
        def code_to_content(code):
            code = ex["starter_code"] + code
            return py_prompt(ex["question"], code)

        def_cols = {
            "question": ex["question"],
            "starter_code": ex[args.code_col],
        }

        for correct in ex[args.code_col]:
            new_ds.append(
                {
                    "content": code_to_content(correct),
                    "score": 1,
                    "solution": correct,
                    "mutation_step": None,
                    **def_cols,
                }
            )
        for mutated, step in zip(ex["mutated"], ex["mutation_step"]):
            new_ds.append(
                {
                    "content": code_to_content(mutated),
                    "score": 0,
                    "solution": mutated,
                    "mutation_step": step,
                    **def_cols,
                }
            )

    new_ds = datasets.Dataset.from_list(new_ds)
    new_ds.save_to_disk(args.output)


if __name__ == "__main__":
    import argparse
    random.seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="codegenning/taco_reasoning_steps_cleaned",
    )
    parser.add_argument(
        "--code_col",
        type=str,
        default="reasoning_steps",
    )
    parser.add_argument(
        "--executor",
        type=str,
        default="http://127.0.0.1:8000",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count(),
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
    )
    args = parser.parse_args()
    main(args)
