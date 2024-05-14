from codeprm.eval.generic import EvaluationManager, get_generic_argparser, make_items_from_ds
import json
from codeprm.model import model_factory
import datasets


def main(args):
    model = model_factory(
        args.model_kind,
        args.model,
        num_gpus=args.num_gpus,
    )
    dataset = datasets.load_dataset(args.dataset, split="test")
    # convert dataset to list
    dataset = dataset.to_list()
    # json loads all tests
    for i, item in enumerate(dataset):
        dataset[i]["input_output"] = json.loads(item["input_output"])

    items = make_items_from_ds(
        dataset,
        "question",
        "input_output",
        starter_code_col="starter_code",
        unique_name_col=None,
        random_sample=args.random_sample,
    )
    manager = EvaluationManager(
        model=model,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        temperature=args.temperature,
        batch_size=args.batch_size,
        completion_limit=args.completion_limit,
        dataset_name=args.dataset,
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
