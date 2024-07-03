from coderm.eval.generic import get_native_coderm_argparser, make_items_from_ds, generic_eval_main
from coderm.model import BaseModel
from typing import Optional
import json
import datasets


def main(args, model: Optional[BaseModel] = None):
    dataset = datasets.load_dataset(args.dataset, split=args.split)
    # convert dataset to list
    dataset = dataset.to_list()
    # json loads all tests
    for i, item in enumerate(dataset):
        dataset[i]["input_output"] = json.loads(item["input_output"])
        if "public_input_output" in item:
            dataset[i]["public_input_output"] = json.loads(
                item["public_input_output"])

    assert len(dataset)

    items = make_items_from_ds(
        dataset,
        "question",
        "input_output",
        public_tests_col="public_input_output",
        starter_code_col="starter_code",
        difficulty_col="difficulty",
        random_sample=args.random_sample,
        unique_name_col="id",
        solutions_col="solutions" if "solutions" in dataset[0] else None
    )
    generic_eval_main(
        args,
        items,
        model=model,
        default_timeout=60,
    )


if __name__ == "__main__":
    parser = get_native_coderm_argparser(
        "codegenning/livecodebench_lite_filtered")
    args = parser.parse_args()
    main(args)
