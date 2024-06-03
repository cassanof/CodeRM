from typing import List
import json
import datasets
import os
from codeprm.eval.generic import EvaluationManager
from codeprm.model import BaseModel, Completion
from codeprm.prompts import Prompt


def get_err_kind(r):
    o = r["output"]
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
    dataset = datasets.load_dataset(args.dataset, split=args.split)
    # convert dataset to list
    dataset = dataset.to_list()
    # json loads all tests
    for i, item in enumerate(dataset):
        dataset[i]["input_output"] = json.loads(item["input_output"])

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
    raise NotImplementedError("TODO!")


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
    parser.add_argument("--exec_batch_size", type=int, default=os.cpu_count())
    parser.add_argument("--redo_kinds", type=str,
                        default="Timeout,Failed,Exception")
    parser.add_argument("--executor", type=str,
                        default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=int,
                        default=60, help="Default timeout in seconds")
    args = parser.parse_args()
