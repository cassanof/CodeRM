from codeprm.eval.generic import EvaluationManager, get_generic_argparser, make_items_from_ds, partition_items
import json
from codeprm.model import model_factory
import datasets


def main(args):
    model = model_factory(
        args.model_kind,
        args.model,
        num_gpus=args.num_gpus,
    )
    dataset = datasets.load_dataset(args.dataset, split=args.split)
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
        difficulty_col="difficulty",
        random_sample=args.random_sample,
        unique_name_col="id",
    )
    items = partition_items(
        items, start_idx=args.start_idx, max_items=args.max_items)
    manager = EvaluationManager(
        model=model,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        temperature=args.temperature,
        completion_limit=args.completion_limit,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        exec_batch_size=args.exec_batch_size,
        timeout=8,  # 6 is what is set by LiveCodeBench. we run with 8 to be safe
    )
    # generate
    manager.generate_completions(items, use_tqdm=True)
    # evaluate
    manager.evaluate_completions(items, use_tqdm=True)
    # save
    manager.save_completions(items, args.output, fmt=args.output_format)


if __name__ == "__main__":
    parser = get_generic_argparser("cassanof/livecodebench_lite_filtered")
    args = parser.parse_args()
    main(args)
