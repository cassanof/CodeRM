from typing import Any, Dict, List, Optional, Tuple
import time
from codeprm.code_exec_server.code_exec_reqs import check_executor_alive
import shutil
from pathlib import Path
from tqdm import tqdm
from codeprm.execution import parse_time_limit, smart_exec_tests, smart_exec_tests_queuebatched
from codeprm.prompts import Prompt
from codeprm.utils import chunkify, gunzip_json_write, gunzip_json_read
from codeprm.model import BaseModel, Completion
import os
import json
from codeprm.model import model_factory


def read_completions_from_disk(path: str) -> Optional[List[Dict[str, Any]]]:
    # either datasets or gzjson
    path_p = Path(path)
    if not path_p.exists():
        return None
    if "gz" in path_p.suffix:
        obj = gunzip_json_read(path_p)
        if obj is None:
            return None
        return obj["items"]
    elif path_p.is_dir():
        import datasets
        ds = datasets.load_from_disk(path)
        return ds.to_list()  # type: ignore


class CompletionResult:
    def __init__(self, passing: bool, output: str):
        self.passing = passing
        self.output = output

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passing": self.passing,
            "output": self.output,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "CompletionResult":
        assert "passing" in d, "Missing 'passing' key"
        assert "output" in d, "Missing 'output' key"
        return CompletionResult(
            d["passing"],
            d["output"],
        )


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

        self.completions: List[Completion] = []
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
        if len(self.completions) == len(self.results):
            results = [{**c.to_dict(), **r.to_dict()}
                       for c, r in zip(self.completions, self.results)]
        elif len(self.results) == 0:  # no exec yet
            results = [{**c.to_dict()} for c in self.completions]
        else:
            raise ValueError("Completions and results don't match")

        return {
            "unique_name": self.unique_name,
            "prompt": self.get_prompt(),
            "starter_code": self.get_starter_code(),
            "difficulty": self.get_difficulty(),
            #  "tests": self.get_tests(), # don't include this in the output, too large
            "results": results,
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


def partition_items(items: List[CompletionItem], start_idx: Optional[int], max_items: Optional[int]) -> List[CompletionItem]:
    # warning if start_idx is not None and max_items is None and vice versa
    if (start_idx is not None and max_items is None) or (start_idx is None and max_items is not None):
        print("WARNING: start_idx and max_items should be used together for parallel processing. Ignoring them.")
    if start_idx is None or max_items is None:
        return items
    return items[start_idx:start_idx + max_items]


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
            timeout=30,
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
        self.timeout = timeout

    def generate_completions(
            self,
            items: List[CompletionItem],
            use_tqdm=True,
            save_every_batch=None,
    ):
        """
        Generates completions for each item in the list and saves them to the item.
        use_tqdm: whether to show progress bar
        save_every_batch: either None, path, or (path, type). If path is given, saves the completions to disk every batch. 
        If type is given, saves in that format. Either "gzjson" or "datasets"
        """
        indexed_prompts: List[Tuple[int, Prompt]] = []

        save_path = None
        save_fmt = None
        if save_every_batch is not None:
            if isinstance(save_every_batch, str):
                save_path = save_every_batch
                save_fmt = "gzjson"
            else:
                save_path, save_fmt = save_every_batch

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
            completions = self.model.generate_with_info(
                prompts,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                temperature=self.temperature,
                use_tqdm=use_tqdm,
            )
            for i, completion in zip(indices, completions):
                items[i].completions.append(completion)

            if save_path is not None and save_fmt is not None:
                self.save_completions(
                    items, save_path, fmt=save_fmt, verbose=False)

    def evaluate_completions(self, items: List[CompletionItem], use_tqdm=True):
        indexed_completions: List[Tuple[int, str]] = []
        for i, item in enumerate(items):
            for completion in item.completions:
                indexed_completions.append((i, completion.code))

        if self.model.prefix_starter_code():
            codes = [items[i].get_starter_code() + completion for i,
                     completion in indexed_completions]
        else:
            codes = [completion for _, completion in indexed_completions]

        tests_per_code = [
            items[i].get_tests() for i, _ in indexed_completions]
        time_limits = [items[i].get_timeout(
            default=self.timeout) for i, _ in indexed_completions]
        results = smart_exec_tests_queuebatched(
            codes,
            tests_per_code,
            timeouts=time_limits,
            executor=self.executor,
            workers=self.exec_batch_size,
            use_tqdm=use_tqdm,
        )

        strugglers = []
        for (i, c), (passing, output) in zip(indexed_completions, results):
            if not passing and "Failed to execute program:" in output:
                strugglers.append((i, c))
            else:
                items[i].results.append(CompletionResult(passing, output))

        if strugglers:
            # wait for container to be alive (if not already)
            max_retries = 5
            for _ in range(max_retries):
                if check_executor_alive(self.executor):
                    break
                time.sleep(1)

            # retry strugglers one by one
            if use_tqdm:
                strugglers = tqdm(
                    strugglers,
                    total=len(strugglers),
                    desc="Container runtime failed, retrying one by one",
                )

            for i, c in strugglers:
                result = smart_exec_tests(
                    items[i].get_starter_code() + c,
                    items[i].get_tests(),
                    timeout=items[i].get_timeout(default=self.timeout),
                    executor=self.executor,
                )
                passing, output = result
                items[i].results.append(CompletionResult(passing, output))

        return results

    def save_completions(
            self,
            items: List[CompletionItem],
            output_path: str,
            fmt="gzjson",
            verbose=True,
    ):
        if fmt == "gzjson":
            outpath = Path(output_path + ".json.gz")
            if outpath.exists():
                outpath.unlink()
            outpath.parent.mkdir(parents=True, exist_ok=True)
            if verbose:
                print(f"Saving completions to {outpath}")
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
        elif fmt == "datasets":
            import datasets
            ds = datasets.Dataset.from_list([item.to_dict() for item in items])
            outpath = Path(output_path)
            if outpath.exists():
                # delete directory, unlink doesn't work for directories
                shutil.rmtree(outpath)

            ds.save_to_disk(output_path)
        else:
            raise ValueError(f"Unknown format {fmt}")

    @staticmethod
    def load_completions(items: List[CompletionItem], path: str):
        completions = read_completions_from_disk(path)
        if completions is None:
            raise ValueError(f"Couldn't read completions from {path}")
        for i, item in enumerate(items):
            item.completions = [Completion.from_dict(c)
                                for c in completions[i]["results"]]
            if "passing" in completions[i]["results"][0]:
                assert all(
                    "passing" in c for c in completions[i]["results"]), "Some completions are missing 'passing' key"
                item.results = [CompletionResult.from_dict(
                    c) for c in completions[i]["results"]]


def generic_eval_main(
        args,
        base_items: List[CompletionItem],
        default_timeout=30,
):
    model = model_factory(
        args.model_kind,
        args.model,
        num_gpus=args.num_gpus,
    )
    items = partition_items(
        base_items, start_idx=args.start_idx, max_items=args.max_items)
    manager = EvaluationManager(
        model=model,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        temperature=args.temperature,
        completion_limit=args.completion_limit,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        exec_batch_size=args.exec_batch_size,
        timeout=default_timeout,
    )

    save_every_batch = None
    if args.save_every_batch:
        save_every_batch = (args.output, args.output_format)

    # generate
    manager.generate_completions(
        items,
        use_tqdm=True,
        save_every_batch=save_every_batch,
    )
    # save before exec
    manager.save_completions(items, args.output, fmt=args.output_format)
    # clean GPU memory, model not needed anymore
    model.free_memory()
    # evaluate
    manager.evaluate_completions(items, use_tqdm=True)
    # save after exec
    manager.save_completions(items, args.output, fmt=args.output_format)


def get_generic_argparser(dataset_default: str, split="test"):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default=dataset_default,
        help="Dataset name"
    )
    parser.add_argument(
        "--split",
        type=str,
        default=split,
        help="Dataset split"
    )
    parser.add_argument(
        "--completion-limit",
        type=int,
        default=1,
        help="Number of completions to generate per problem"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1028,
        help="Total batch size for generation"
    )
    cpu_count = os.cpu_count()
    if cpu_count is None:
        cpu_count = 1
    parser.add_argument(
        "--exec-batch-size",
        type=int,
        default=cpu_count,
        help="Total batch size for execution (defaults to os.cpu_count())"
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
        choices=["base", "few-shot", "openai"]
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
        default=0.0,
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
    parser.add_argument(
        "--output-format",
        type=str,
        default="gzjson",
        help="Output format. either 'gzjson' or 'datasets'",
        choices=["gzjson", "datasets"]
    )
    parser.add_argument(
        "--save-every-batch",
        action="store_true",
        help="Save completions every batch. Useful for long running processes"
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=None,
        help="Start index for the dataset. Useful for parallel processing in combination with --max-items",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Max items to process. Useful for parallel processing in combination with --start-idx",
    )
    return parser
