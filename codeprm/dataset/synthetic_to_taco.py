import datasets


def main(args):
    ds = datasets.load_from_disk(args.input)
    new_ds = []
    for d in ds:
        solutions = []
        for r in d["results"]:
            if r["passing"]:
                code = r["code"]
                solutions.append(code)

        new_ds.append({
            "question": d["prompt"],
            "difficulty": d["difficulty"],
            "unique_name": d["unique_name"],
            "solutions": solutions
        })

    new_ds = datasets.Dataset.from_list(new_ds)
    new_ds = new_ds.filter(lambda x: len(x["solutions"]) > 0)
    print(new_ds)
    total_solns = sum([len(x["solutions"]) for x in new_ds])
    print(f"Total solutions: {total_solns}")
    new_ds.save_to_disk(args.output)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(
        description='Takes a completion dataset and makes it into a taco-format dataset with the solution and prompt cols')
    parser.add_argument('--input', type=str,
                        help='path to the input dataset', required=True)
    parser.add_argument('--output', type=str,
                        help='path to the output dataset', required=True)
    args = parser.parse_args()
    main(args)
