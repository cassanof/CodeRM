from typing import List, Union
import copy
import json
import datasets
import os
from coderm.eval.generic import CompletionResult, EvaluationManager, make_items_from_ds, start_anti_congestion_routine
from coderm.model import BaseModel, Completion
from coderm.prompts import Prompt


def get_err_kind(r: Union[dict, CompletionResult]) -> str:
    if isinstance(r, dict):
        o = r["output"]
    else:  # must be CompletionResult
        o = r.output

    if "Timeout" in o:
        t = "Timeout"
    elif "expected" in o and "but got" in o or "AssertionError" in o:
        t = "Failed"
    elif "Failed to execute program:" in o:
        t = "Error"
    else:
        t = "Exception"
    return t


class MockModel(BaseModel):
    def __init__(self, prefix_starter_code=True):
        super().__init__("mock")
        self.do_prefix = prefix_starter_code

    def generate_with_info(self, prompts: List[Prompt], **kwargs) -> List[Completion]:
        raise NotImplementedError()

    def format_prompt(self, question: str, code="") -> Prompt:
        raise NotImplementedError()

    def prefix_starter_code(self) -> bool:
        return self.do_prefix


def main(args):
    dataset = datasets.load_dataset(args.dataset, split=args.split)
    # convert dataset to list
    dataset = dataset.to_list()
    first_item = dataset[0]

    # different cols to support different datasets
    test_col = "input_output" if "input_output" in first_item else "test"
    starter_code_col = "starter_code" if "starter_code" in first_item else None
    id_col = "task_id" if "task_id" in first_item else "id"
    difficulty_col = "difficulty" if "difficulty" in first_item else None
    public_tests_col = "public_input_output" if "public_input_output" in first_item else None

    # json loads all tests
    for i, og_item in enumerate(dataset):
        try:
            dataset[i][test_col] = json.loads(og_item[test_col])
            if "public_input_output" in og_item:
                dataset[i]["public_input_output"] = json.loads(
                    og_item["public_input_output"])
        except json.JSONDecodeError:
            continue

    manager = EvaluationManager(
        model=MockModel(not args.no_prefix_starter_code),
        max_tokens=0,
        top_p=0.0,
        temperature=0.0,
        batch_size=1,
        executor=args.executor,
        exec_batch_size=args.exec_batch_size,
        dataset_name=args.dataset,
        completion_limit=1,
        timeout=args.timeout,
    )

    def make_items():
        return make_items_from_ds(
            dataset,
            "question",
            test_col,
            public_tests_col=public_tests_col,
            starter_code_col=starter_code_col,
            difficulty_col=difficulty_col,
            random_sample=None,
            unique_name_col=id_col,
        )

    og_items = make_items()
    redo_items = make_items()
    manager.load_completions(og_items, args.input)
    print(f"Loaded {len(og_items)} items")

    # 1. select items to redo, with reduced completion lists
    assert len(og_items) == len(redo_items)

    for og_item, redo_item in zip(og_items, redo_items):
        og_item = copy.deepcopy(og_item)
        if args.redo == "all":
            redo_item.completions = og_item.completions
        elif args.redo in ["failed", "timeout", "error"]:
            # if "failed", redo all, if "timeout", redo only timeouts
            completions_to_redo = []
            for c, r in zip(og_item.completions, og_item.results):
                if not r.passing:
                    if args.redo == "failed" or \
                            (args.redo == "timeout" and get_err_kind(r) == "Timeout") or \
                            (args.redo == "error" and get_err_kind(r) == "Error"):
                        completions_to_redo.append(c)
            redo_item.completions = completions_to_redo

    # edge case: if len(og_items[...].results) == 0, then the eval was not done. fill with None
    if any(len(og_item.results) == 0 for og_item in og_items):
        print("Warning: some items were not evaluated in the original completion file. Filling with None and forcing --redo to 'all'")
        for og_item in og_items:
            og_item.results = [None] * len(og_item.completions)  # type: ignore

        args.redo = "all"

    if args.anti_congestion:
        start_anti_congestion_routine()

    # 2. redo completions
    manager.evaluate_completions(redo_items, exec_public=args.exec_public)

    assert len(og_items) == len(redo_items)
    # 3. merge in redone completions to original items
    for og_item, redo_item in zip(og_items, redo_items):
        completion_to_res = {}
        assert len(redo_item.completions) == len(redo_item.results)
        for c, r in zip(redo_item.completions, redo_item.results):
            completion_to_res[c.code] = r

        for i, c in enumerate(og_item.completions):
            if c.code in completion_to_res:
                og_item.results[i] = completion_to_res[c.code]

    # 4. save output
    manager.save_completions(
        og_items,
        args.output,
        fmt="gzjson" if args.input.endswith("json.gz") else "datasets",
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="Path to the input completion file")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to the eval dataset")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split to use")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to the output completion file")
    parser.add_argument("--no-prefix-starter-code", action="store_true")
    parser.add_argument("--anti-congestion", action="store_true")
    parser.add_argument("--exec-public", action="store_true")
    cpu_count = os.cpu_count()
    if cpu_count is None:
        cpu_count = 1
    else:
        cpu_count = int(cpu_count * 0.8)  # lower for stability
    parser.add_argument("--exec-batch-size", type=int, default=cpu_count)
    parser.add_argument(
        "--redo", type=str, choices=["failed", "timeout", "error", "all"], default="error")
    parser.add_argument("--executor", type=str,
                        default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=int,
                        default=60, help="Default timeout in seconds")
    args = parser.parse_args()
    main(args)
