from typing import Any, Dict, List, Optional
from tqdm import tqdm
from codeprm.utils import chunkify
from codeprm.model import BaseModel


class CompletionItem:
    def __init__(self, prompt_col: str, item: Dict[str, Any], starter_code_col: Optional[str] = None):
        self.prompt_col = prompt_col
        self.starter_code = starter_code_col
        self.item = item
        self.completions = []

    def get_prompt(self) -> str:
        return self.item[self.prompt_col]

    def get_starter_code(self) -> Optional[str]:
        if self.starter_code is not None:
            return self.item[self.starter_code]
        return None


def make_items_from_ds(dataset, prompt_col: str, starter_code_col: Optional[str] = None) -> List[CompletionItem]:
    items = []
    for item in dataset:
        items.append(
            CompletionItem(
                prompt_col,
                item,
                starter_code_col=starter_code_col,
            )
        )
    return items


class CompletionGenerator:
    def __init__(
            self,
            model: BaseModel,
            max_tokens: int,
            top_p: float,
            temperature: float,
            batch_size: int,
            completion_limit: int,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.temperature = temperature
        self.batch_size = batch_size
        self.completion_limit = completion_limit

    def generate_completions(self, items: List[CompletionItem], use_tqdm=True):
        indexed_prompts = []

        for i, example in enumerate(items):
            indexed_prompts.extend(
                [(i, example.get_prompt())] * self.completion_limit)

        chunks = chunkify(indexed_prompts, self.batch_size)

        if use_tqdm:
            chunks = tqdm(chunks, total=len(chunks),
                          desc="Generating batches of completions")

        for chunk in chunks:
            indices, prompts = zip(*chunk)
            completions = self.model.generate(
                prompts,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                temperature=self.temperature,
            )
            for i, completion in zip(indices, completions):
                items[i].completions.append(completion)
