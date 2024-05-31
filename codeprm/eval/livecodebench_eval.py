from codeprm.eval.generic import get_generic_argparser, make_items_from_ds, generic_eval_main
import json
import datasets


def main(args):
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
    generic_eval_main(
        args,
        items,
        default_timeout=60,
    )


if __name__ == "__main__":
    parser = get_generic_argparser("cassanof/livecodebench_lite_filtered")
    args = parser.parse_args()
    main(args)
