from typing import Tuple
import datasets
import re


def extract_docstring(code: str, entry_point: str) -> Tuple[str, str]:
    """
    Extracts the docstring from a code snippet. Returns the code snippet without the docstring and the extracted docstring.
    """
    # this is used when there are more than one function in the code
    # invariant: the function pointed to by entry_point is the last one in the code
    to_search = code.split(f"def {entry_point}")[1]
    docstring = re.search(r'(?s)"""(.*?)"""', to_search)
    if docstring is None:
        return code, ""
    docstring = docstring.group(0)
    code = code.replace(docstring, "").strip()
    docstring = docstring.strip().strip('"""').strip()
    # remove 4 spaces of indentation from docstring
    docstring = docstring.replace("\n    ", "\n")
    code = code + "\n"
    return code, docstring


def map_fn(ex):
    test = ex["test"] + f"\n\ncheck({ex['entry_point']})"
    code, docstring = extract_docstring(ex["prompt"], ex["entry_point"])
    return {
        "question": docstring,
        "starter_code": code,
        "test": test,
    }


def main(args):
    ds = datasets.load_dataset(args.dataset, split='test')
    ds = ds.map(map_fn)
    ds.push_to_hub(args.push, split='test')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        default="evalplus/humanevalplus")
    parser.add_argument('--push', type=str, required=True)
    args = parser.parse_args()
    main(args)
