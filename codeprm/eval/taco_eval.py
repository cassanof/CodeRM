from codeprm.eval.generic import CompletionManager, make_items_from_ds
from codeprm.execution import smart_exec_tests_batched
from codeprm.model import model_factory
import datasets


def main(args):
    model = model_factory(
        args.model_kind,
        args.model,
        num_gpus=args.num_gpus,
    )
    dataset = datasets.load_dataset(args.dataset, split="test")

    items = make_items_from_ds(
        dataset,
        "question",
        "input_output",
        starter_code_col="starter_code",
        unique_name_col="url",
    )
    manager = CompletionManager(
        model,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        temperature=args.temperature,
        batch_size=args.batch_size,
        completion_limit=args.completion_limit,
    )
    # generate
    manager.generate_completions(items, use_tqdm=True)
    # evaluate
    manager.evaluate_completions(items, use_tqdm=True)

    # evaluate


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="cassanof/taco_cleaned_eval",
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
        "--output-dir",
        type=str,
        required=True,
        help="Output directory to store completions"
    )
    args = parser.parse_args()
    main(args)
