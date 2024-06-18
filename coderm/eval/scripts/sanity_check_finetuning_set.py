import datasets
import ast


def does_parse(code):
    try:
        ast.parse(code)
        return True
    except SyntaxError as e:
        print(f"SyntaxError: {e}")
        return False


def main(args):
    dataset = datasets.load_dataset(args.dataset, split="train")
    for ex in dataset:
        code = ex["solutions"].replace(args.mark, "")
        if not does_parse(code):
            print(f"Code:\n{code}")
            break


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--mark', type=str, default="<issue_start>")
    args = parser.parse_args()
    main(args)
