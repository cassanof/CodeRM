import os
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import threading
import torch
from codeprm.prompts import py_prompt, py_prompt_2shot_lcb, py_prompt_2shot_lcb_chat, Conversation, Prompt
from codeprm.utils import markdown_codeblock_extract

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
    elif kind == "openai":
        return OpenAIChatModel(name, prompt_fn=py_prompt_2shot_lcb_chat)
    else:
        raise ValueError(f"Unknown model kind: {kind}")


class BaseModel(ABC):
    # init method
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def generate_with_info(self, prompts: List[Prompt], **kwargs) -> List[Completion]:
        pass

    @abstractmethod
    def format_prompt(self, question: str, code="") -> Prompt:
        pass

    def get_name(self) -> str:
        return self.model_name

    def generate(self, prompts: List[Prompt], **kwargs) -> List[str]:
        completions = self.generate_with_info(prompts, **kwargs)
        return [c.code for c in completions]

    def prefix_starter_code(self) -> bool:
        # whether the model requires to prefix starter code to responses
        # chat models do not require this
        return True


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
            # max_model_len=4096,
        )
        self.prompt_fn = prompt_fn

    def generate_with_info(self, prompts: List[Prompt], **kwargs) -> List[Completion]:
        from vllm import SamplingParams
        kwargs = kwargs.copy()
        stop = kwargs.pop("stop", [])
        stop.append("# START NEW CODE")  # for few-shot prompts
        gens = self.model.generate(
            prompts=prompts,
            sampling_params=SamplingParams(
                top_p=kwargs.pop("top_p", 1.0),
                temperature=kwargs.pop("temperature", 0.0),
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

    def format_prompt(self, question: str, code="") -> str:
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


def post_process_markdown(new: str) -> str:
    try:
        extracted = markdown_codeblock_extract(new)
    except Exception as e:
        print(f"Failed to extract codeblock from {new}: {e}")
        extracted = new
    return extracted.strip()


def logprobs_to_cumulative(logprobs):  # NOTE: normalized
    c = 0
    for l in logprobs:
        c += l
    return c / len(logprobs)


class OpenAIChatModel(BaseModel):
    def __init__(self, model_name: str, prompt_fn=py_prompt_2shot_lcb_chat):
        super().__init__(model_name)
        from openai import Client
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("OPENAI_API_KEY is not set")
        self.client = Client(api_key=api_key)
        self.prompt_fn = prompt_fn

    def generate_with_info(self, prompts: List[Prompt], **kwargs) -> List[Completion]:
        completions: List[Optional[Completion]] = [None] * len(prompts)
        threads = []

        def generate_completion(prompt, i):
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=prompt,
                max_tokens=kwargs.get("max_tokens", 3076),
                logprobs=True,
                stop=kwargs.get("stop", []),
                temperature=kwargs.get("temperature", 0.0),
                top_p=kwargs.get("top_p", 1.0),
            )
            choice = response.choices[0]
            o = choice.message.content
            logprobs = choice.logprobs.content  # type: ignore
            assert o is not None, "OpenAI returned a null response"
            assert logprobs is not None, "OpenAI returned a null logprobs"
            logprobs = [l.logprob for l in logprobs]
            num_tokens = len(logprobs)
            proc = post_process_markdown(o)
            cumulative_logprob = logprobs_to_cumulative(logprobs)
            completions[i] = Completion(
                proc, cumulative_logprob, num_tokens)

        for i, prompt in enumerate(prompts):
            thread = threading.Thread(
                target=generate_completion, args=(prompt, i))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert all(
            c is not None for c in completions), "Some completions are missing -- threading bug?"

        return completions  # type: ignore

    def format_prompt(self, question: str, code="") -> Conversation:
        return self.prompt_fn(question, code)

    def prefix_starter_code(self) -> bool:
        return False
