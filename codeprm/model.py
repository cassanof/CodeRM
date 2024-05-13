from typing import List
import torch

from abc import ABC, abstractmethod


def autodetect_dtype() -> str:
    if torch.cuda.is_bf16_supported():
        return "bfloat16"
    else:
        return "auto"


def py_prompt(question: str, code=""):
    # escape any triple quotes in the question
    question = question.replace('"""', r'\"""')
    return f'''"""
{question}
"""
{code}'''


class BaseModel(ABC):
    @abstractmethod
    def generate(self, prompts: List[str], **kwargs) -> List[str]:
        pass


class HFModel(BaseModel):
    def __init__(self, model_name: str, num_gpus=1):
        from vllm import LLM
        self.model = LLM(model_name, tensor_parallel_size=num_gpus)

    def generate(self, prompts: List[str], **kwargs) -> List[str]:
        from vllm import SamplingParams
        kwargs = kwargs.copy()
        gens = self.model.generate(
            prompts=prompts,
            sampling_params=SamplingParams(
                top_p=kwargs.pop("top_p", 0.9),
                temperature=kwargs.pop("temperature", 0.2),
                max_tokens=kwargs.pop("max_tokens", 2048),
            )
            use_tqdm=kwargs.pop("use_tqdm", False),
        )
        return [gen.outputs[0].text for gen in gens]
