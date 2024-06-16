import multiprocessing
import torch
from generic import MakeShiftWandbCallback, convert_2_label_rm_to_1_label_rm

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)

from trl import ModelConfig
from trl.trainer.rloo_trainer import RLOOConfig, RLOOTrainer
from trl.trainer.utils import SIMPLE_QUERY_CHAT_TEMPLATE
from transformers.trainer_callback import PrinterCallback, DefaultFlowCallback
from dataclasses import dataclass


@dataclass
class Args:
    train_dataset: str = "codegenning/taco-rl"
    """the dataset to train on"""
    test_dataset: str = "codegenning/taco-rl"
    """the dataset to evaluate on"""
    train_split: str = "train"
    """the split to train on"""
    test_split: str = "test"
    """the split to evaluate on"""

    num_labels: int = 1
    """the number of labels in the reward model. 1 or 2 supported"""
    pos_idx: int = 1
    """if using two labels, the index of the positive label"""

    max_prompt_length: int = 2048
    """the maximum length of the prompt"""


if __name__ == "__main__":
    parser = HfArgumentParser((Args, RLOOConfig, ModelConfig))
    args, rloo_config, model_config = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        padding_side="left",
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_QUERY_CHAT_TEMPLATE

    reward_model = AutoModelForSequenceClassification.from_pretrained(
        rloo_config.reward_model_path, num_labels=args.num_labels, use_flash_attention_2=True, torch_dtype=torch.float16)
    if args.num_labels == 2:
        reward_model = convert_2_label_rm_to_1_label_rm(reward_model)
    elif args.num_labels != 1:
        raise ValueError(
            f"Only binary classification or regression is supported, got {args.num_labels} labels")

    ref_policy = AutoModelForCausalLM.from_pretrained(
        rloo_config.sft_model_path, use_flash_attention_2=True, torch_dtype=torch.float16)
    policy = AutoModelForCausalLM.from_pretrained(rloo_config.sft_model_path, use_flash_attention_2=True, torch_dtype=torch.float16)

    train_dataset = load_dataset(args.train_dataset, split=args.train_split)
    eval_dataset = load_dataset(args.test_dataset, split=args.test_split)

    def prepare_dataset(dataset, tokenizer):
        """pre-tokenize the dataset before training; only collate during training"""

        def tokenize(element):
            input_ids = tokenizer.encode(
                element["prompt"],
                padding=False,
                add_special_tokens=True,
            )
            return {"input_ids": input_ids, "lengths": len(input_ids)}

        return dataset.map(
            tokenize,
            remove_columns=dataset.column_names,
            num_proc=1 if rloo_config.sanity_check else multiprocessing.cpu_count(),
            load_from_cache_file=not rloo_config.sanity_check,
        )

    train_dataset = prepare_dataset(train_dataset, tokenizer)
    eval_dataset = prepare_dataset(eval_dataset, tokenizer)

    train_dataset = train_dataset.filter(
        lambda x: x["lengths"] <= args.max_prompt_length)
    eval_dataset = eval_dataset.filter(
        lambda x: x["lengths"] <= args.max_prompt_length)
    assert train_dataset[0]["input_ids"][-1] != tokenizer.eos_token_id, "The last token should not be an EOS token"

    trainer = RLOOTrainer(
        config=rloo_config,
        tokenizer=tokenizer,
        policy=policy,
        ref_policy=ref_policy,
        reward_model=reward_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[
            DefaultFlowCallback,
            PrinterCallback,
            MakeShiftWandbCallback,
        ],
    )
    trainer.train()
    trainer.save_model(rloo_config.output_dir)
    trainer.push_to_hub()
    trainer.generate_completions()
