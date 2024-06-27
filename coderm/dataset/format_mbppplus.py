import datasets

def map_fn(ex):
    # get code without soln
    lines = []
    code = ex["code"]
    defcount = code.count("def ")
    curcount = 0
    for l in code.split("\n"):
        lines.append(l)
        if l.startswith("def "):
            curcount += 1
            if curcount == defcount:
                break
    else:
        raise ValueError("Could not find last def")
    code = "\n".join(lines)
    return {
        "question": ex["prompt"],
        "starter_code": code,
    }


def main(args):
    ds = datasets.load_dataset(args.dataset, split='test')
    ds = ds.map(map_fn)
    ds.push_to_hub(args.push, split='test')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        default="evalplus/mbppplus")
    parser.add_argument('--push', type=str, required=True)
    args = parser.parse_args()
    main(args)
