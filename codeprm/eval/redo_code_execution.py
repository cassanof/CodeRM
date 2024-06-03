from typing import List, Union
import copy
import json
import datasets
import os
from codeprm.eval.generic import CompletionResult, EvaluationManager, make_items_from_ds, start_anti_congestion_routine
from codeprm.model import BaseModel, Completion
from codeprm.prompts import Prompt


def get_err_kind(r: Union[dict, CompletionResult]) -> str:
    if isinstance(r, dict):
        o = r["output"]
    else:  # must be CompletionResult
        o = r.output

    if "Timeout" in o:
        t = "Timeout"
    elif "expected" in o and "but got" in o or "AssertionError" in o:
        t = "Failed"
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
    assert ".json.gz" not in args.input, "Please provide the completion file in datasets format"
    dataset = datasets.load_dataset(args.dataset, split=args.split)
    # convert dataset to list
    dataset = dataset.to_list()
    # json loads all tests
    for i, og_item in enumerate(dataset):
        dataset[i]["input_output"] = json.loads(og_item["input_output"])

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
            "input_output",
            starter_code_col="starter_code",
            difficulty_col="difficulty",
            random_sample=None,
            unique_name_col=None,
        )

    og_items = make_items()
    redo_items = make_items()
    manager.load_completions(og_items, args.input)
    print(f"Loaded {len(og_items)} items")

    # 1. select items to redo, with reduced completion lists

    for og_item, redo_item in zip(og_items, redo_items):
        og_item = copy.deepcopy(og_item)
        if args.redo == "all":
            redo_item.completions = og_item.completions
        elif args.redo in ["failed", "timeout"]:
            # if "failed", redo all, if "timeout", redo only timeouts
            completions_to_redo = []
            for c, r in zip(og_item.completions, og_item.results):
                if not r.passing:
                    if args.redo == "failed" or (args.redo == "timeout" and get_err_kind(r) == "Timeout"):
                        completions_to_redo.append(c)
            redo_item.completions = completions_to_redo

    # edge case: if len(og_items[...].results) == 0, then the eval was not done. fill with None
    if any(len(og_item.results) == 0 for og_item in og_items):
        print("Warning: some items were not evaluated in the original completion file. Filling with None")
        for og_item in og_items:
            og_item.results = [None] * len(og_item.completions)  # type: ignore

    if args.anti_congestion:
        start_anti_congestion_routine()

    # 2. redo completions
    manager.evaluate_completions(redo_items)

    # 3. merge in redone completions to original items
    for og_item, redo_item in zip(og_items, redo_items):
        completion_to_res = {}
        for c, r in zip(redo_item.completions, redo_item.results):
            completion_to_res[c] = r

        for i, c in enumerate(og_item.completions):
            if c in completion_to_res:
                og_item.results[i] = completion_to_res[c]

    # 4. save output
    manager.save_completions(og_items, args.output, fmt="datasets")


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
    parser.add_argument("--exec-batch-size", type=int, default=os.cpu_count())
    parser.add_argument(
        "--redo", type=str, choices=["failed", "timeout", "all"], default="timeout")
    parser.add_argument("--executor", type=str,
                        default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=int,
                        default=60, help="Default timeout in seconds")
    args = parser.parse_args()
    main(args)
