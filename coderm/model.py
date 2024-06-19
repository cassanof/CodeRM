import os
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import numpy as np
import threading
from coderm.prompts import py_prompt, py_prompt_2shot_lcb, py_prompt_2shot_lcb_chat, Conversation, Prompt, py_prompt_evolve
from coderm.utils import markdown_codeblock_extract
import torch

from abc import ABC, abstractmethod


EVOLVED_SEP = "# ==== EVOLVED CODE ===="


class Completion:
    def __init__(self, code: str, cumulative_logprob: float, num_tokens: int, orm_score: Optional[float] = None):
        self.code = code
        self.cumulative_logprob = cumulative_logprob
        self.num_tokens = num_tokens
        self.orm_score = orm_score

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "code": self.code,
            "cumulative_logprob": self.cumulative_logprob,
            "num_tokens": self.num_tokens,
        }

        if self.orm_score is not None:
            d["orm_score"] = self.orm_score

        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Completion":
        assert "code" in d, "Missing 'code' key"
        assert "cumulative_logprob" in d, "Missing 'cumulative_logprob' key"
        assert "num_tokens" in d, "Missing 'num_tokens' key"
        return Completion(
            d["code"],
            d["cumulative_logprob"],
            d["num_tokens"],
            orm_score=d.get("orm_score", None),
        )


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
        evolver_e=25,
        evolver_t=0.95,
        rm=None,
):
    if kind == "base":
        return HFModel(name, num_gpus=num_gpus, prompt_fn=py_prompt)
    elif kind == "few-shot":
        return HFModel(name, num_gpus=num_gpus, prompt_fn=py_prompt_2shot_lcb)
    elif kind == "openai":
        return OpenAIChatModel(name, prompt_fn=py_prompt_2shot_lcb_chat)
    elif kind == "evolver":
        if rm is None:
            raise ValueError(
                "OutcomeRewardModel is required for evolver model. Set with the 'rm' parameter.")
        return EvolverModel(name, rm, num_gpus=num_gpus, evolver_e=evolver_e, evolver_t=evolver_t)
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

    def free_memory(self):
        pass  # only used for local models

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
    def score(self, contents: List[str]) -> List[Tuple[float, float]]:
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
            dtype=autodetect_dtype_str(),
            max_model_len=16384,
            trust_remote_code=True,
        )
        self.num_gpus = num_gpus
        self.prompt_fn = prompt_fn

    def generate_with_info(self, prompts: List[Prompt], **kwargs) -> List[Completion]:
        from vllm import SamplingParams
        kwargs = kwargs.copy()
        stop = kwargs.pop("stop", [])
        stop.append("# START NEW CODE")  # for few-shot prompts
        stop.append(EVOLVED_SEP)  # for evaluating base evolver
        gens = self.model.generate(
            prompts=prompts,
            sampling_params=SamplingParams(
                top_p=kwargs.pop("top_p", 0.95),
                temperature=kwargs.pop("temperature", 0.0),
                max_tokens=kwargs.pop("max_tokens", 4096),
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

    def free_memory(self):
        from vllm.distributed.parallel_state import destroy_model_parallel
        import torch
        import gc
        destroy_model_parallel()
        del self.model.llm_engine.model_executor.driver_worker
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
        if self.num_gpus > 1:
            # NOTE: unfortunately no way to free main process when TP>1 :(
            import ray
            ray.shutdown()

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
    def __init__(self, model_name: str, device=None, pos_idx=1):
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
        # figure out if it's a 2-label classification or regression model
        if self.model.config.num_labels == 2:
            self.is_classification = True
        else:
            self.is_classification = False
        self.pos_idx = pos_idx

    def score(self, contents: List[str], **kwargs) -> List[Tuple[float, float]]:
        max_length = kwargs.get("max_length", 4096)
        with torch.no_grad():
            inputs = self.tokenizer(
                contents,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(self.device)
            outputs = self.model(**inputs)
            logits = outputs.logits
            scores = []
            if self.is_classification:
                probs = torch.nn.functional.softmax(logits, dim=-1)
                probs = probs.cpu().to(torch.float32).numpy()
                for prob in probs:
                    scores.append((float(prob[0]), float(prob[1])))
            else:
                # model is trained with MSE loss, so we can interpret the output as a probability
                # (since it's a single scalar value), should be in [0, 1]
                probs = logits.cpu().to(torch.float32).numpy()
                for prob in probs:
                    scores.append((float(1 - prob), float(prob)))

            return scores


EvolutionStrategy = Literal["cma", "es", "random"]


class EvolverModel(HFModel):
    def __init__(
        self,
        model_name: str,
        rm_name: str,
        num_gpus=1,
        evolver_e=25,  # maximum number of iterations
        evolver_t=0.95,  # target reward rate. early stopping if reached
        rm_device=None,
        prompt_fn=py_prompt,
        evol_prompt_fn=py_prompt_evolve,
    ):
        super().__init__(model_name, num_gpus=num_gpus, prompt_fn=prompt_fn)
        self.rm = OutcomeRewardModel(rm_name, device=rm_device)
        self.evolver_e = evolver_e
        self.evolver_t = evolver_t
        self.evol_prompt_fn = evol_prompt_fn

    def evolve(self, prompt: Prompt, program: Optional[str], **kwargs) -> Completion:
        """
        Evolves a program for the given task (from the prompt) into another program and 
        scores the completion with the reward model.

        If program is None, the model will generate the program from scratch.
        """
        assert isinstance(
            prompt, str), "Prompt must be a string for evolver model"

        if program is None:
            evol_prompt = prompt
        else:
            evol_prompt = self.evol_prompt_fn(prompt + program)

        completion = super().generate_with_info([evol_prompt], **kwargs)[0]
        to_score = prompt + completion.code
        scores = self.rm.score([to_score])[0][self.rm.pos_idx]
        completion.orm_score = scores
        return completion

    def evol_generate(self, prompt: Prompt, **kwargs) -> Completion:
        pool = []
        for _ in range(self.evolver_e):
            completion = self.evolve(prompt, None, **kwargs)
            pool.append(completion)

            assert completion.orm_score is not None, "ORM score is missing"
            if completion.orm_score >= self.evolver_t:
                break

        best = max(pool, key=lambda c: c.orm_score)
        return best

    def generate_with_info(self, prompts: List[Prompt], **kwargs) -> List[Completion]:
        # TODO: spend a weekend batching this
        outs = []
        for prompt in prompts:
            res = self.evol_generate(prompt, **kwargs)
            outs.append(res)
        return outs

    def format_prompt(self, question: str, code="") -> str:
        return self.prompt_fn(question, code)  # not the evolve one!


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
