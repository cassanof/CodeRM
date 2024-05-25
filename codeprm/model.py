from typing import Any, Dict, List, Tuple
import numpy as np
import torch
from codeprm.prompts import py_prompt, py_prompt_2shot_lcb

from abc import ABC, abstractmethod


class Completion:
    def __init__(self, code: str, cumulative_logprob: float, num_tokens: int):
        self.code = code
        self.cumulative_logprob = cumulative_logprob
        self.num_tokens = num_tokens

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "cumulative_logprob": self.cumulative_logprob,
            "num_tokens": self.num_tokens,
        }


def autodetect_dtype_str() -> str:
    if torch.cuda.is_bf16_supported():
        return "bfloat16"
    else:
        return "auto"


def autodetect_dtype() -> torch.dtype:
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    else:
        return torch.float16


def model_factory(
        kind: str,
        name: str,
        num_gpus=1,
):
    if kind == "base":
        return HFModel(name, num_gpus=num_gpus, prompt_fn=py_prompt)
    elif kind == "few-shot":
        return HFModel(name, num_gpus=num_gpus, prompt_fn=py_prompt_2shot_lcb)
    else:
        raise ValueError(f"Unknown model kind: {kind}")


class BaseModel(ABC):
    # init method
    def __init__(self, model_name: str):
        self.model_name = model_name

    def get_name(self) -> str:
        return self.model_name

    @abstractmethod
    def generate_with_info(self, prompts: List[str], **kwargs) -> List[Completion]:
        pass

    def generate(self, prompts: List[str], **kwargs) -> List[str]:
        completions = self.generate_with_info(prompts, **kwargs)
        return [c.code for c in completions]

    @abstractmethod
    def format_prompt(self, question: str, code=""):
        pass


class ClassificationModel(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def get_name(self) -> str:
        return self.model_name

    @abstractmethod
    def score(self, contents: List[str]) -> List[Tuple[int, float]]:
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

    def generate_with_info(self, prompts: List[str], **kwargs) -> List[Completion]:
        from vllm import SamplingParams
        kwargs = kwargs.copy()
        stop = kwargs.pop("stop", [])
        stop.append("# START NEW CODE")  # for few-shot prompts
        gens = self.model.generate(
            prompts=prompts,
            sampling_params=SamplingParams(
                top_p=kwargs.pop("top_p", 0.9),
                temperature=kwargs.pop("temperature", 0.2),
                max_tokens=kwargs.pop("max_tokens", 2048),
                stop=stop,
            ),
            use_tqdm=kwargs.pop("use_tqdm", False),
        )
        outs = []
        for gen in gens:
            gen = gen.outputs[0]
            outs.append(Completion(
                gen.text,
                gen.cumulative_logprob,
                len(gen.token_ids),
            ))
        return outs

    def format_prompt(self, question: str, code=""):
        return self.prompt_fn(question, code)


def detect_first_unused_device() -> str:
    import torch
    for i in range(torch.cuda.device_count()):
        if not torch.cuda.memory_reserved(i):
            return f"cuda:{i}"
    print("WARNING: No available GPU detected, using CPU. Specify --device to use a specific device.")
    return "cpu"


class OutcomeRewardModel(ClassificationModel):
    def __init__(self, model_name: str, device=None):
        super().__init__(model_name)
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        if device is None:
            device = detect_first_unused_device()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, add_eos_token=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=autodetect_dtype(),
            use_flash_attention_2=True,
            use_cache=False,
        ).to(self.device).eval()

    def score(self, contents: List[str], **kwargs) -> List[Tuple[int, float]]:
        max_length = kwargs.get("max_length", 4096)
        with torch.no_grad():
            inputs = self.tokenizer(
                contents,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(self.device)
            # goal of this function is to return class id and probability of the class
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            probs = probs.cpu().to(torch.float32).numpy()
            scores = []
            for i in range(len(probs)):
                score = probs[i]
                scores.append((int(np.argmax(score)), float(np.max(score))))
            return scores
