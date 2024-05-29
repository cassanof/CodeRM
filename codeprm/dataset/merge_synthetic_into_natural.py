from sys import meta_path
import datasets


def question_dedup(ds):
    qset = set()
    new_ds = []
    for ex in ds:
        q = ex["question"]
        if q not in qset:
            new_ds.append(ex)
            qset.add(q)
    return datasets.Dataset.from_list(new_ds)


def main(args):
    natural = datasets.load_from_disk(args.natural)
    synthetic = datasets.load_from_disk(args.synthetic)
    print("Natural:", len(natural))
    print("Synthetic:", len(synthetic))

    # dedup based on question
    natural = question_dedup(natural)
    synthetic = question_dedup(synthetic)

    print("Natural (dedup):", len(natural))
    print("Synthetic (dedup):", len(synthetic))

    # merge same question from synthetic into natural
    q2ex = {}  # question -> example
    for ex in synthetic:
        q = ex["question"]
        q2ex[q] = ex

    for ex in natural:
        q = ex["question"]
        if q in q2ex:
            # merge solutions
            q2ex[q][args.synthetic_soln_col].extend(ex[args.natural_soln_col])
        else:
            if args.synthetic_soln_col != args.natural_soln_col:
                ex[args.synthetic_soln_col] = ex[args.natural_soln_col]
                del ex[args.natural_soln_col]
            q2ex[q] = ex

    merged = list(q2ex.values())
    print("Merged:", len(merged))
    total_solutions = sum(len(ex[args.merged_col]) for ex in merged)
    print("Total solutions:", total_solutions)
    merged_ds = datasets.Dataset.from_list(merged)
    if not args.merged_col == args.synthetic_soln_col:
        merged_ds = merged_ds.rename_column(args.synthetic_soln_col, args.merged_col)
    merged_ds.save_to_disk(args.output)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--natural", type=str, required=True)
    parser.add_argument("--synthetic", type=str, required=True)
    parser.add_argument("--synthetic_soln_col", type=str, default="solutions")
    parser.add_argument("--natural_soln_col", type=str,
                        default="reasoning_steps")
    parser.add_argument("--merged_col", type=str, default="solutions")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    main(args)
