from coderm.eval.generic import get_native_coderm_argparser, make_items_from_ds, generic_eval_main
from coderm.model import BaseModel
import json
import datasets
from typing import Optional


def main(args,  model: Optional[BaseModel] = None):
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
        unique_name_col=None,
    )
    generic_eval_main(
        args,
        items,
        model=model,
        default_timeout=30,
    )


if __name__ == "__main__":
    parser = get_native_coderm_argparser(
        "cassanof/taco_cleaned_all", split="train")
    args = parser.parse_args()
    main(args)
