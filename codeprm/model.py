from typing import List
import torch
from codeprm.prompts import py_prompt

from abc import ABC, abstractmethod


def autodetect_dtype() -> str:
    if torch.cuda.is_bf16_supported():
        return "bfloat16"
    else:
        return "auto"


def model_factory(
        kind: str,
        name: str,
        num_gpus=1,
):
    if kind == "base":
        return HFModel(name, num_gpus=num_gpus)
    else:
        raise ValueError(f"Unknown model kind: {kind}")


class BaseModel(ABC):
    # init method
    def __init__(self, model_name: str):
        self.model_name = model_name

    def get_name(self) -> str:
        return self.model_name

    @abstractmethod
    def generate(self, prompts: List[str], **kwargs) -> List[str]:
        pass

    @abstractmethod
    def format_prompt(self, question: str, code=""):
        pass


class HFModel(BaseModel):
    def __init__(
            self,
            model_name: str,
            num_gpus=1,
            prompt_fn=py_prompt,
    ):
        super().__init__(model_name)
        from vllm import LLM
        self.model = LLM(
            model_name,
            tensor_parallel_size=num_gpus,
            enforce_eager=True,
            max_model_len=4096,
        )
        self.prompt_fn = prompt_fn

    def generate(self, prompts: List[str], **kwargs) -> List[str]:
        from vllm import SamplingParams
        kwargs = kwargs.copy()
        gens = self.model.generate(
            prompts=prompts,
            sampling_params=SamplingParams(
                top_p=kwargs.pop("top_p", 0.9),
                temperature=kwargs.pop("temperature", 0.2),
                max_tokens=kwargs.pop("max_tokens", 2048),
            ),
            use_tqdm=kwargs.pop("use_tqdm", False),
        )
        return [gen.outputs[0].text for gen in gens]

    def format_prompt(self, question: str, code=""):
        return self.prompt_fn(question, code)
