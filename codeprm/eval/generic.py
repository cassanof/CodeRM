from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from tqdm import tqdm
from codeprm.execution import parse_time_limit, smart_exec_tests_batched
from codeprm.utils import chunkify, gunzip_json_write
from codeprm.model import BaseModel
import os


class CompletionResult:
    def __init__(self, passing: bool, output: str):
        self.passing = passing
        self.output = output

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passing": self.passing,
            "output": self.output,
        }


class CompletionItem:
    def __init__(
            self,
            unique_name: str,
            prompt_col: str,
            tests_col: str,
            item: Dict[str, Any],
            starter_code_col: Optional[str] = None,
            difficulty_col: Optional[str] = None,
    ):
        self.prompt_col = prompt_col
        self.starter_code_col = starter_code_col
        self.tests_col = tests_col
        self.item = item
        self.unique_name = unique_name
        self.difficulty_col = difficulty_col

        self.completions: List[str] = []
        self.results: List[CompletionResult] = []

    def get_prompt(self) -> str:
        return self.item[self.prompt_col]

    def get_tests(self) -> Any:  # TODO: proper types
        return self.item[self.tests_col]

    def get_difficulty(self) -> Optional[str]:
        if self.difficulty_col is not None:
            return self.item[self.difficulty_col]
        return None

    def get_timeout(self, default=30) -> int:
        if "time_limit" in self.item:
            return parse_time_limit(self.item["time_limit"], default=default)
        return default

    def get_starter_code(self) -> str:
        if self.starter_code_col is not None:
            starter = self.item[self.starter_code_col]
            starter = starter if starter is not None else ""
            return starter
        return ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unique_name": self.unique_name,
            "prompt": self.get_prompt(),
            "starter_code": self.get_starter_code(),
            "difficulty": self.get_difficulty(),
            #  "tests": self.get_tests(), # don't include this in the output, too large
            "results": [{"code": c, **r.to_dict()} for c, r in zip(self.completions, self.results)],
        }


def make_items_from_ds(
        dataset,
        prompt_col: str,
        tests_col: str,
        difficulty_col: Optional[str] = None,
        starter_code_col: Optional[str] = None,
        random_sample: Optional[int] = None,
        unique_name_col: Optional[str] = None,
) -> List[CompletionItem]:
    # these are replaced with "_" if used in unique_name
    bad_chars = ["\n", "\r", "\t", "/", " "]

    if unique_name_col is not None:
        # check that it's actually unique
        ids = set()
        for item in dataset:
            assert item[unique_name_col] not in ids, "Unique name column is not actually unique"
            ids.add(item[unique_name_col])

    if random_sample is not None:
        if isinstance(dataset, list):
            import random
            random.seed(42)
            dataset = random.sample(dataset, random_sample)
        else:
            dataset = dataset.shuffle(seed=42).select(range(random_sample))

    items = []
    for i, item in enumerate(dataset):
        unique_name = str(i)
        if unique_name_col is not None:
            unique_name = item[unique_name_col]
            for char in bad_chars:
                unique_name = unique_name.replace(char, "_")

        items.append(
            CompletionItem(
                unique_name,
                prompt_col,
                tests_col,
                item,
                starter_code_col=starter_code_col,
                difficulty_col=difficulty_col,
            )
        )

    return items


class EvaluationManager:
    def __init__(
            self,
            model: BaseModel,
            max_tokens: int,
            top_p: float,
            temperature: float,
            batch_size: int,
            completion_limit: int,
            dataset_name: str,
            exec_batch_size=os.cpu_count(),
            executor="http://127.0.0.1:8000",
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.temperature = temperature
        self.completion_limit = completion_limit
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.exec_batch_size = exec_batch_size if exec_batch_size is not None else 1
        self.executor = executor

    def generate_completions(self, items: List[CompletionItem], use_tqdm=True):
        indexed_prompts = []

        for i, example in enumerate(items):
            indexed_prompts.extend(
                [(i, self.model.format_prompt(example.get_prompt(), example.get_starter_code()))] * self.completion_limit)

        chunks = chunkify(indexed_prompts, self.batch_size)

        if use_tqdm:
            chunks = tqdm(
                chunks,
                total=len(chunks),
                desc="Generating batches of completions (batch size:"
                + f" {self.batch_size})",
            )

        for chunk in chunks:
            indices, prompts = zip(*chunk)
            completions = self.model.generate(
                prompts,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                temperature=self.temperature,
                use_tqdm=use_tqdm,
            )
            for i, completion in zip(indices, completions):
                items[i].completions.append(completion)

    def evaluate_completions(self, items: List[CompletionItem], use_tqdm=True):
        indexed_completions: List[Tuple[int, str]] = []
        for i, item in enumerate(items):
            for completion in item.completions:
                indexed_completions.append((i, completion))

        chunks = chunkify(indexed_completions, self.exec_batch_size)

        if use_tqdm:
            chunks = tqdm(
                chunks,
                total=len(chunks),
                desc="Executing batches of completions (batch size:"
                + f" {self.exec_batch_size})",
            )

        for chunk in chunks:
            codes = [items[i].get_starter_code() + completion for i,
                     completion in chunk]
            tests_per_code = [
                items[i].get_tests() for i, _ in chunk]
            time_limits = [items[i].get_timeout() for i, _ in chunk]
            results = smart_exec_tests_batched(
                codes,
                tests_per_code,
                timeouts=time_limits,
                executor=self.executor,
            )
            for (i, _), (passing, output) in zip(chunk, results):
                items[i].results.append(CompletionResult(passing, output))

    def save_completions(self, items: List[CompletionItem], output_path: str, verbose=True):
        outpath = Path(output_path + ".json.gz")
        if verbose:
            print(f"Saving completions to {outpath}...")
        d = {
            "model": self.model.get_name(),
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "completion_limit": self.completion_limit,
            "dataset_name": self.dataset_name,
            "items": [item.to_dict() for item in items],
        }
        gunzip_json_write(outpath, d)


def get_generic_argparser(dataset_default: str):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default=dataset_default,
        help="Dataset name"
    )
    parser.add_argument(
        "--completion-limit",
        type=int,
        default=20,
        help="Number of completions to generate per problem"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1028,
        help="Total batch size for generation"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Max new tokens in the generated text"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model to use"
    )
    parser.add_argument(
        "--model-kind",
        type=str,
        default="base",
        help="Model kind",
        choices=["base"]
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Temperature for sampling. Set to 0 for greedy decoding"
    )
    parser.add_argument(
        "--random-sample",
        type=int,
        default=None,
        help="Randomly (seed=42) sample this many examples from the dataset and evaluate. By default, None, so evaluates the entire dataset"
    )
    parser.add_argument(
        "--executor",
        type=str,
        default="http://127.0.0.1:8000",
        help="Server URL for executing the code"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path to store the results. don't add extension"
    )
    return parser
