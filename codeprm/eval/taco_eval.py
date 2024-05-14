from codeprm.eval.generic import EvaluationManager, get_generic_argparser, make_items_from_ds
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
    manager = EvaluationManager(
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
    # save
    manager.save_completions(items, args.output)


if __name__ == "__main__":
    parser = get_generic_argparser("cassanof/taco_cleaned_eval")
    args = parser.parse_args()
    main(args)
