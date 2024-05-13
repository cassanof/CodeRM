from codeprm.eval.generic import CompletionGenerator, make_items_from_ds
from codeprm.model import HFModel
import datasets


def main(args):
    model = HFModel(args.model, num_gpus=args.num_gpus)
    dataset = datasets.load_dataset(args.dataset, split="test")

    items = make_items_from_ds(
        dataset,
        "question",
        starter_code_col="starter_code",
    )
    CompletionGenerator(
        model,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        temperature=args.temperature,
        batch_size=args.batch_size,
        completion_limit=args.completion_limit,
    ).generate_completions(items, use_tqdm=True)


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
        "--output-dir",
        type=str,
        required=True,
        help="Output directory to store completions"
    )
    args = parser.parse_args()
    main(args)
