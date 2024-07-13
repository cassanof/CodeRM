from coderm.prompts import py_prompt
import random
import datasets


def main(args):
    random.seed(42)
    ds = datasets.load_from_disk(args.input)

    new_ds = []
    for ex in ds:
        chosen = None
        rejected = None

        for r in ex["results"]:
            code = r["code"]
            if r["passing"]:
                chosen = code
            else:
                rejected = code

        if chosen is None or rejected is None:
            continue

        starter = ex["starter_code"]
        prompt = py_prompt(ex["prompt"], starter)
        if starter is None or starter == "":
            prompt += "\n"

        defs = {
            "prompt": prompt,
            "text_chosen": chosen,
            "text_rejected": rejected,
        }
        new_ds.append(defs)

    final_ds = datasets.Dataset.from_list(new_ds)
    final_ds.push_to_hub(args.push, private=True, split=args.push_split)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="Input dataset path for the results")
    parser.add_argument("--push", type=str, required=True,
                        help="Push dataset path for the ORM")
    parser.add_argument("--push-split", type=str, default="train",
                        help="The split for the pushed dataset")
    args = parser.parse_args()
    main(args)
