from typing import Optional
from coderm.eval.generic import get_native_coderm_argparser, make_items_from_ds, generic_eval_main
from coderm.model import BaseModel
import json
import datasets


def main(args, model: Optional[BaseModel] = None):
    dataset = datasets.load_dataset(args.dataset, split=args.split).to_list()
    items = make_items_from_ds(
        dataset,
        "question",
        "test",
        starter_code_col="starter_code",
        difficulty_col=None,
        random_sample=args.random_sample,
        unique_name_col="task_id",
    )
    generic_eval_main(
        args,
        items,
        model=model,
        default_timeout=120,  # lots of tests....
    )


if __name__ == "__main__":
    parser = get_native_coderm_argparser("cassanof/humanevalplus_formatted")
    args = parser.parse_args()
    main(args)
