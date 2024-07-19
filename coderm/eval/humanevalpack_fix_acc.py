from typing import Optional, Tuple
from tqdm import tqdm
from coderm.model import OutcomeRewardModel
import datasets


def format_ex(ex) -> Tuple[str, str]:  # passing, failing
    p = f'''"""\n{ex["instruction"]}\n"""\n{ex["declaration"]}'''
    buggy = p + f'{ex["buggy_solution"]}'
    correct = p + f'{ex["canonical_solution"]}'
    return correct, buggy


def model_factory(kind: str, name: str):
    if kind == "orm":
        return OutcomeRewardModel(name)
    else:
        raise ValueError(f"Unknown model kind: {kind}")


def predicted_passing(model, kind, code) -> bool:
    if kind == "orm":
        neg, pos = model.score([code])[0]
        return pos > neg
    else:
        raise ValueError(f"Unknown model kind: {kind}")


def main(args):
    ds = datasets.load_dataset(
        "bigcode/humanevalpack", "python", split="test", trust_remote_code=True)
    bug_types = set(ds["bug_type"])

    model = model_factory(args.model_kind, args.model)

    tp = 0
    # track which bug types are being misclassified
    fp = {bt: 0 for bt in bug_types}
    # track which ones are good!
    tn = {bt: 0 for bt in bug_types}
    fn = 0

    def update(code, bug: Optional[str]):
        nonlocal tp
        nonlocal fn
        pred = predicted_passing(model, args.model_kind, code)
        if pred:
            if bug is None:
                tp += 1
            else:
                fp[bug] += 1
        else:
            if bug is None:
                fn += 1
            else:
                tn[bug] += 1

    for ex in tqdm(ds, total=len(ds)):
        c, b = format_ex(ex)
        update(c, None)
        update(b, ex["bug_type"])

    print(f"TP: ({tp}) {tp / len(ds)}")
    print(f"FN: {fn}")
    print(f"TN: ({sum(tn.values())}) {sum(tn.values()) / len(ds)} - {tn}")
    print(f"FP: ({sum(fp.values())}) - {fp}")
    print(
        f"Accuracy: {(tp + sum(tn.values())) / (tp + sum(tn.values()) + sum(fp.values()) + fn)}")
    # per-bug type accuracy
    print("--- Per-bug type accuracy ---")
    for bt in bug_types:
        print(f"{bt}: {tn[bt] / (tn[bt] + fp[bt])}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-kind", type=str,
                        default="orm", choices=["orm"])
    parser.add_argument("--model", required=True, type=str)
    args = parser.parse_args()
    main(args)
