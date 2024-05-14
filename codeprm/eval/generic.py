from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm
from codeprm.execution import parse_time_limit, smart_exec_tests_batched
from codeprm.utils import chunkify
from codeprm.model import BaseModel
import os


class CompletionResult:
    def __init__(self, passing: bool, output: str):
        self.passing = passing
        self.output = output


class CompletionItem:
    def __init__(
            self,
            unique_name: str,
            prompt_col: str,
            tests_col: str,
            item: Dict[str, Any],
            starter_code_col: Optional[str] = None,
    ):
        self.prompt_col = prompt_col
        self.starter_code_col = starter_code_col
        self.tests_col = tests_col
        self.item = item
        self.unique_name = unique_name

        self.completions: List[str] = []
        self.results: List[CompletionResult] = []

    def get_prompt(self) -> str:
        return self.item[self.prompt_col]

    def get_tests(self) -> Any:  # TODO: proper types
        return self.item[self.tests_col]

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


def make_items_from_ds(
        dataset,
        prompt_col: str,
        tests_col: str,
        starter_code_col: Optional[str] = None,
        random_sample: Optional[int] = None,
        unique_name_col: Optional[str] = None,
) -> List[CompletionItem]:
    # these are replaced with "_" if used in unique_name
    bad_chars = ["\n", "\r", "\t", "/", " "]

    if unique_name_col is not None:
        # check that it's actually unique
        assert len(set(dataset[unique_name_col])) == len(
            dataset), "Unique name column is not actually unique"

    if random_sample is not None:
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
            )
        )

    return items


class CompletionManager:
    def __init__(
            self,
            model: BaseModel,
            max_tokens: int,
            top_p: float,
            temperature: float,
            batch_size: int,
            completion_limit: int,
            exec_batch_size=os.cpu_count(),
            executor="http://127.0.0.1:8000",
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.temperature = temperature
        self.completion_limit = completion_limit
        self.batch_size = batch_size
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
                desc="Generating batches of completions",
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
                desc="Evaluating completions",
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
